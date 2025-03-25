import cv2
import numpy as np
import time
from collections import deque
import os
import sys
import torch

# import logging
# logging.getLogger("ultralytics").setLevel(logging.WARNING)  # Only show warnings and errors

# Dictionary mapping common objects to degradable/non-degradable status
DEGRADABILITY_MAP = {
    # Degradable items
    'apple': 'degradable',
    'banana': 'degradable',
    'orange': 'degradable',
    'paper': 'degradable',
    'cardboard': 'degradable',
    'plant': 'degradable',
    'flower': 'degradable',
    'leaf': 'degradable',
    'wood': 'degradable',
    'cotton': 'degradable',
    'food': 'degradable',
    'book': 'degradable',
    'sandwich': 'degradable',
    'hot dog': 'degradable',
    'pizza': 'degradable',
    'cake': 'degradable',
    'donut': 'degradable',
    'orange': 'degradable',
    'broccoli': 'degradable',
    'carrot': 'degradable',
    'fruit': 'degradable',
    'vegetable': 'degradable',
    'apple': 'degradable',
    'potted plant': 'degradable',
    'teddy bear': 'degradable',  # Usually made of cloth/cotton
    
    # Non-degradable items
    'bottle': 'non-degradable',
    'plastic': 'non-degradable',
    'cell phone': 'non-degradable',
    'laptop': 'non-degradable',
    'remote': 'non-degradable',
    'metal': 'non-degradable',
    'can': 'non-degradable',
    'glass': 'non-degradable',
    'ceramic': 'non-degradable',
    'television': 'non-degradable',
    'cup': 'non-degradable',
    'fork': 'non-degradable',
    'knife': 'non-degradable',
    'spoon': 'non-degradable',
    'bowl': 'non-degradable',
    'chair': 'non-degradable',
    'couch': 'non-degradable',
    'bed': 'non-degradable',
    'dining table': 'non-degradable',
    'toilet': 'non-degradable',
    'tv': 'non-degradable',
    'laptop': 'non-degradable',
    'mouse': 'non-degradable',
    'keyboard': 'non-degradable',
    'cell phone': 'non-degradable',
    'microwave': 'non-degradable',
    'oven': 'non-degradable',
    'toaster': 'non-degradable',
    'sink': 'non-degradable',
    'refrigerator': 'non-degradable',
    'car': 'non-degradable',
    'bicycle': 'non-degradable',
    'motorcycle': 'non-degradable',
    'truck': 'non-degradable',
    'boat': 'non-degradable',
    'airplane': 'non-degradable',
    'bus': 'non-degradable',
    'train': 'non-degradable',
    'traffic light': 'non-degradable',
    'fire hydrant': 'non-degradable',
    'stop sign': 'non-degradable',
    'parking meter': 'non-degradable',
    'bench': 'non-degradable',
    'backpack': 'non-degradable',
    'umbrella': 'non-degradable',
    'handbag': 'non-degradable',
    'tie': 'non-degradable',
    'suitcase': 'non-degradable',
    'frisbee': 'non-degradable',
    'skis': 'non-degradable',
    'snowboard': 'non-degradable',
    'sports ball': 'non-degradable',
    'kite': 'non-degradable',
    'baseball bat': 'non-degradable',
    'baseball glove': 'non-degradable',
    'skateboard': 'non-degradable',
    'surfboard': 'non-degradable',
    'tennis racket': 'non-degradable',
    'vase': 'non-degradable',
    'scissors': 'non-degradable',
    'toothbrush': 'non-degradable',
    'hair drier': 'non-degradable',
}

class YOLOv11Detector:
    def __init__(self, model_size='n'):
        """
        Initialize YOLOv11 detector
        
        Args:
            model_size (str): Model size - 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        """
        # For prediction stabilization
        self.prediction_history = deque(maxlen=5)
        self.last_stable_prediction = None
        self.stable_count = 0
        self.min_stable_frames = 3
        
        # Model configuration
        self.model_size = model_size
        
        # Try to import ultralytics and initialize model
        try:
            self.install_requirements()
            import ultralytics
            from ultralytics import YOLO
            
            # Load YOLOv11 model
            model_path = f"yolo11{model_size}.pt"
            # Check if model exists locally
            if not os.path.exists(model_path):
                print(f"Downloading YOLOv11-{model_size} model...")
                # If not, we need to download it from ultralytics or fall back to YOLOv8
                try:
                    # Try to download YOLOv11 model
                    self.model = YOLO(model_path)
                    print(f"YOLOv11-{model_size} model loaded successfully")
                except Exception as e:
                    print(f"Failed to download YOLOv11 model: {e}")
                    print("Falling back to YOLOv8 model")
                    self.model = YOLO(f"yolov8{model_size}.pt")
                    print(f"YOLOv8-{model_size} model loaded successfully")
            else:
                # Load existing model
                self.model = YOLO(model_path)
                print(f"YOLOv11-{model_size} model loaded successfully")
            
        except ImportError as e:
            print(f"Error: {e}")
            print("Could not import required packages.")
            sys.exit(1)
            
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def install_requirements(self):
        """Install required packages if they're not already installed"""
        try:
            import ultralytics
        except ImportError:
            print("Installing ultralytics package...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
            print("Ultralytics installed successfully")
    
    def detect_objects(self, frame, conf_threshold=0.25):
        """
        Detect objects using YOLOv11
        
        Args:
            frame: Input image frame
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of detection results with bounding boxes
        """
        # Run inference
        results = self.model(frame, conf=conf_threshold, verbose=False)[0]
        
        # Process detections
        detections = []
        
        for det in results.boxes.data:
            x1, y1, x2, y2, confidence, class_id = det
            x1, y1, x2, y2, class_id = map(int, [x1, y1, x2, y2, class_id])
            
            # Get class label
            class_name = results.names[class_id]
            
            # Check if the object is in our degradability map
            degradability = DEGRADABILITY_MAP.get(class_name.lower(), 'unknown')
            factor = 100
            detections.append({
                'object': class_name,
                'confidence': float(confidence),
                'degradability': degradability,
                'box': (x1-factor, y1-factor, x2-factor, y2-factor)
            })
            
        return detections
    
    def stabilize_predictions(self, results):
        """
        Stabilize predictions to prevent jumping between classes
        """
        # If no results, return empty
        if not results:
            return []
            
        # Add current top prediction to history
        if results:
            self.prediction_history.append(results[0]['object'])
        
        # Count occurrences of each prediction in history
        from collections import Counter
        prediction_counts = Counter(self.prediction_history)
        
        # Get the most common prediction
        most_common = prediction_counts.most_common(1)[0]
        most_common_label, count = most_common
        
        # If the most common prediction is stable enough, use it
        if count >= self.min_stable_frames:
            self.last_stable_prediction = most_common_label
            self.stable_count = count
        
        # Filter results to only include the stable prediction
        if self.last_stable_prediction:
            stable_results = [r for r in results if r['object'] == self.last_stable_prediction]
            if stable_results:
                return stable_results
        
        # If no stable prediction yet, return the original results
        return results
    
    def detect_from_camera(self, enable_stabilization=True, conf_threshold=0.25):
        """
        Real-time detection from webcam with YOLOv11
        
        Args:
            enable_stabilization: Whether to stabilize predictions
            conf_threshold: Confidence threshold for detections
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        print("Press 'q' to quit")
        
        # For FPS calculation
        prev_time = 0
        fps_history = deque(maxlen=10)
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture image")
                break
            
            # Get the current time for FPS calculation
            current_time = time.time()
            if prev_time > 0:
                current_fps = 1 / (current_time - prev_time)
                fps_history.append(current_fps)
                fps = sum(fps_history) / len(fps_history)  # Average FPS for stability
            else:
                fps = 0
            prev_time = current_time
            
            # Run object detection
            results = self.detect_objects(frame, conf_threshold)
            
            # Apply stabilization if requested
            if enable_stabilization and results:
                results = self.stabilize_predictions(results)
            
            # Draw results on frame
            for result in results:
                # Get bounding box
                x1, y1, x2, y2 = result['box']
                
                # Set color based on degradability
                if result['degradability'] == 'degradable':
                    color = (0, 255, 0)  # Green (BGR)
                elif result['degradability'] == 'non-degradable':
                    color = (0, 0, 255)  # Red (BGR)
                else:
                    color = (255, 255, 0)  # Yellow (BGR)
                
                # Draw bounding box with rounded corners
                self.draw_rounded_rectangle(frame, (x1, y1), (x2, y2), color, 2, 10)
                
                # Create label text
                text = f"{result['object']} ({result['confidence']:.2f}): {result['degradability']}"
                
                # Calculate text size for background rectangle
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw background rectangle for text
                cv2.rectangle(frame, 
                             (x1, y1 - 25), 
                             (x1 + text_size[0], y1), 
                             color, -1)
                
                # Display text
                cv2.putText(frame, text, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Display model info - get actual model type from the model object
            model_type = f"YOLOv11-{self.model_size}" if "11" in str(self.model) else f"YOLOv8-{self.model_size}"
            cv2.putText(frame, f"Model: {model_type}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show the frame
            cv2.imshow('YOLO Degradability Detector', frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    def draw_rounded_rectangle(self, img, pt1, pt2, color, thickness, radius):
        """Draw a rectangle with rounded corners"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Draw main rectangle without corners
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # Draw the four corners
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

# Example usage
if __name__ == "__main__":
    try:
        # Choose model size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        # Smaller models are faster but less accurate, larger models are more accurate but slower
        model_size = 'n'  # Nano is the fastest option for real-time detection
        
        print("Initializing YOLOv11 Degradability Detector...")
        print(f"Using model size: {model_size} (nano)")
        print("This will automatically download the model if needed (~6MB for nano)")
        
        # Create and initialize the detector
        detector = YOLOv11Detector(model_size=model_size)
        
        # Run detection with webcam
        detector.detect_from_camera(enable_stabilization=True, conf_threshold=0.3)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()