import cv2
import numpy as np
import os
import urllib.request
import sys
import time

# COCO dataset classes that YOLOv3 was trained on
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Dictionary mapping each class to degradable/non-degradable category
DEGRADABILITY_MAPPING = {
    # Degradable items (organic or biodegradable)
    'person': 'degradable',
    'bird': 'degradable',
    'cat': 'degradable',
    'dog': 'degradable',
    'horse': 'degradable',
    'sheep': 'degradable',
    'cow': 'degradable',
    'elephant': 'degradable',
    'bear': 'degradable',
    'zebra': 'degradable',
    'giraffe': 'degradable',
    'banana': 'degradable',
    'apple': 'degradable',
    'sandwich': 'degradable',
    'orange': 'degradable',
    'broccoli': 'degradable',
    'carrot': 'degradable',
    'hot dog': 'degradable',
    'pizza': 'degradable',
    'donut': 'degradable',
    'cake': 'degradable',
    'potted plant': 'degradable',
    'book': 'degradable',  # paper is degradable
    
    # Non-degradable items (synthetic materials, metals, etc.)
    'bicycle': 'non-degradable',
    'car': 'non-degradable',
    'motorcycle': 'non-degradable',
    'airplane': 'non-degradable',
    'bus': 'non-degradable',
    'train': 'non-degradable',
    'truck': 'non-degradable',
    'boat': 'non-degradable',
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
    'bottle': 'non-degradable',  # assuming plastic/glass
    'wine glass': 'non-degradable',
    'cup': 'non-degradable',  # assuming ceramic/plastic
    'fork': 'non-degradable',
    'knife': 'non-degradable',
    'spoon': 'non-degradable',
    'bowl': 'non-degradable',  # assuming ceramic/plastic
    'chair': 'non-degradable',
    'couch': 'non-degradable',
    'bed': 'non-degradable',
    'dining table': 'non-degradable',
    'toilet': 'non-degradable',
    'tv': 'non-degradable',
    'laptop': 'non-degradable',
    'mouse': 'non-degradable',
    'remote': 'non-degradable',
    'keyboard': 'non-degradable',
    'cell phone': 'non-degradable',
    'microwave': 'non-degradable',
    'oven': 'non-degradable',
    'toaster': 'non-degradable',
    'sink': 'non-degradable',
    'refrigerator': 'non-degradable',
    'clock': 'non-degradable',
    'vase': 'non-degradable',
    'scissors': 'non-degradable',
    'teddy bear': 'non-degradable',
    'hair drier': 'non-degradable',
    'toothbrush': 'non-degradable'
}

def download_yolo_files():
    """Download YOLOv3-tiny model files (faster for real-time)"""
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Using YOLOv3-tiny which is much faster for real-time applications
    weights_url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
    config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
    
    weights_path = os.path.join(models_dir, "yolov3-tiny.weights")
    config_path = os.path.join(models_dir, "yolov3-tiny.cfg")
    
    # Download config file if needed
    if not os.path.exists(config_path):
        print(f"Downloading YOLO config file...")
        urllib.request.urlretrieve(config_url, config_path)
        print("Downloaded YOLO config file")
    
    # Download weights file if needed
    if not os.path.exists(weights_path):
        print(f"Downloading YOLO weights file (this may take a moment)...")
        urllib.request.urlretrieve(weights_url, weights_path)
        print("Downloaded YOLO weights file")
    
    return config_path, weights_path

class RealTimeDegradabilityClassifier:
    def __init__(self, weights_path, config_path):
        # Check if files exist before loading
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        # Load YOLO network
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        
        # Explicitly set to CPU backend and target
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Using CPU backend")
        
        # Get output layer names
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        self.classes = COCO_CLASSES
        self.degradability_mapping = DEGRADABILITY_MAPPING
    
    def process_frame(self, frame, confidence_threshold=0.1, nms_threshold=0.4):
        # Get frame dimensions
        height, width, _ = frame.shape
        
        # Create a blob from the frame and perform a forward pass
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Run forward pass
        outputs = self.net.forward(self.output_layers)
        
        # Initialize lists for detected boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []
        
        # Process each output layer
        for output in outputs:
            for detection in output:
                # Extract class scores
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > confidence_threshold:
                    # Scale bounding box coordinates to the original frame
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression to eliminate redundant overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        
        # Prepare results
        results = []
        
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                class_name = self.classes[class_ids[i]]
                degradability = self.degradability_mapping.get(class_name, 'unknown')
                
                results.append({
                    'class_name': class_name,
                    'degradability': degradability,
                    'confidence': confidences[i],
                    'box': (x, y, x + w, y + h)
                })
                
        return results
    
    def draw_results(self, frame, results):
        # Draw bounding boxes and labels on the frame
        for result in results:
            startX, startY, endX, endY = result['box']
            
            # Set color based on degradability
            if result['degradability'] == 'degradable':
                color = (0, 255, 0)  # Green for degradable (BGR format)
            elif result['degradability'] == 'non-degradable':
                color = (0, 0, 255)  # Red for non-degradable (BGR format)
            else:
                color = (0, 165, 255)  # Orange for unknown (BGR format)
            
            # Draw bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
            # Prepare label text
            label = f"{result['class_name']}: {result['degradability']} ({result['confidence']:.2f})"
            
            # Calculate text size for better positioning
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Ensure label background is within frame boundaries
            y_position = max(startY - 10, label_height + 10)
            
            # Draw label background
            cv2.rectangle(
                frame, 
                (startX, y_position - label_height - 10), 
                (startX + label_width, y_position + baseline - 10), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame, 
                label, 
                (startX, y_position - 7), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                2
            )
        
        # Add summary statistics
        degradable_count = sum(1 for r in results if r['degradability'] == 'degradable')
        non_degradable_count = sum(1 for r in results if r['degradability'] == 'non-degradable')
        
        # Draw statistics at the top of the frame
        cv2.putText(
            frame, 
            f"Degradable: {degradable_count} | Non-degradable: {non_degradable_count}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        return frame
    
    def run_realtime_detection(self, camera_id=-1, window_name="Degradability Classifier"):
        # Initialize video capture from webcam
        cap = cv2.VideoCapture(camera_id)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        print("Starting real-time detection...")
        print("Press 'q' to quit, 's' to save a screenshot")
        
        fps_counter = 0
        fps_start_time = time.time()
        fps = 0
        
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting...")
                break
            
            # Process frame
            results = self.process_frame(frame)
            
            # Draw results on frame
            frame_with_results = self.draw_results(frame.copy(), results)
            
            # Calculate FPS
            fps_counter += 1
            if (time.time() - fps_start_time) > 1:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            # Display FPS
            cv2.putText(
                frame_with_results, 
                f"FPS: {fps}", 
                (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
            
            # Display frame
            cv2.imshow(window_name, frame_with_results)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            
            # Quit on 'q' key
            if key == ord('q'):
                break
                
            # Save screenshot on 's' key
            elif key == ord('s'):
                screenshot_path = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(screenshot_path, frame_with_results)
                print(f"Screenshot saved as {screenshot_path}")
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Real-time detection stopped.")

# Run the program
if __name__ == "__main__":
    try:
        # Download YOLO model files
        print("Setting up YOLO model...")
        config_path, weights_path = download_yolo_files()
        
        # Create classifier instance
        classifier = RealTimeDegradabilityClassifier(
            weights_path=weights_path,
            config_path=config_path
        )
        
        # Choose camera device (0 is usually the default webcam)
        camera_id = 0
        if len(sys.argv) > 1 and sys.argv[1].isdigit():
            camera_id = int(sys.argv[1])
        
        # Run real-time detection
        classifier.run_realtime_detection(camera_id=camera_id)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure all required files are available.")
    except Exception as e:
        print(f"An error occurred: {e}")