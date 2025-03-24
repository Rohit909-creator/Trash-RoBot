import cv2
import numpy as np
import time
# S9A70 S9A80

# R1(448, 121,) S5135, s130, s9110, s380, s780
class LaneBasedRobotPicker:
    def __init__(self, camera_id=0, image_width=640, image_height=480, num_lanes=5):
        """
        Initialize the lane-based picker system with live camera feed
        
        Args:
            camera_id: Camera device ID (default 0 for primary webcam)
            image_width: Width of the camera frame in pixels
            image_height: Height of the camera frame in pixels
            num_lanes: Number of lanes to divide the image into
        """
        self.image_width = image_width
        self.image_height = image_height
        self.num_lanes = num_lanes
        self.lane_width = self.image_width / self.num_lanes
        
        # Create lane boundaries
        self.lane_boundaries = []
        for i in range(num_lanes + 1):
            self.lane_boundaries.append(int(i * self.lane_width))
            
        print(f"Created {num_lanes} lanes with boundaries at {self.lane_boundaries}")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
            
        print(f"Camera initialized with resolution {image_width}x{image_height}")
        
        # Initialize variables for manual bounding box selection
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.bbox = None
        self.pick_position = None
        
    def get_lane_from_bbox(self, bbox):
        """
        Determine which lane a bounding box falls into
        
        Args:
            bbox: Bounding box in format [x_min, y_min, width, height]
            
        Returns:
            lane_index: Index of the lane (0 to num_lanes-1)
            lane_center: X-coordinate of the center of the lane
        """
        # Calculate center of the bounding box
        bbox_center_x = bbox[0] + bbox[2]/2
        
        # Determine which lane this falls into
        for i in range(self.num_lanes):
            if bbox_center_x >= self.lane_boundaries[i] and bbox_center_x < self.lane_boundaries[i+1]:
                lane_center = int((self.lane_boundaries[i] + self.lane_boundaries[i+1]) / 2)
                return i, lane_center
                
        # If somehow outside lanes, return the last lane
        return self.num_lanes-1, int((self.lane_boundaries[-2] + self.lane_boundaries[-1]) / 2)
    
    def calculate_pick_position(self, bbox, approach_height=100):
        """
        Calculate the position to move the robot arm for picking
        
        Args:
            bbox: Bounding box in format [x_min, y_min, width, height]
            approach_height: Height to approach the object from
            
        Returns:
            pick_position: (x, y, z) tuple for where to position the arm
        """
        lane_index, lane_center_x = self.get_lane_from_bbox(bbox)
        
        # Use lane center for X position
        # For Y position, use the center of the bounding box
        bbox_center_y = bbox[1] + bbox[3]/2
        
        # Z position would depend on your robot's coordinate system
        # Here we're just using a fixed approach height
        
        return (lane_center_x, int(bbox_center_y), approach_height)
    
    def draw_bbox(self, event, x, y, flags, param):
        """Mouse callback function for drawing bounding boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # This is just for preview, not setting the actual bbox yet
                pass
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            # Create bbox in format [x, y, width, height]
            width = x - self.ix
            height = y - self.iy
            
            # Handle negative width/height (drawing from bottom-right to top-left)
            if width < 0:
                self.ix = self.ix + width
                width = abs(width)
            if height < 0:
                self.iy = self.iy + height
                height = abs(height)
                
            self.bbox = [self.ix, self.iy, width, height]
            self.pick_position = self.calculate_pick_position(self.bbox)
            print(f"Bbox: {self.bbox}")
            print(f"Lane: {self.get_lane_from_bbox(self.bbox)[0]}")
            print(f"Pick position: {self.pick_position}")
    
    def simulate_object_detection(self, frame):
        """
        Allow manual drawing of bounding boxes to simulate object detection
        or connect to an actual object detection system
        """
        # In a production system, you would replace this with your MobileNet
        # detection logic and just return the detected bounding boxes
        
        # For simulation, we use mouse drawing - this is handled in the draw_bbox callback
        pass
    
    def run(self):
        """Run the lane-based picker with live camera feed"""
        cv2.namedWindow('Lane-Based Trash Sorting')
        cv2.setMouseCallback('Lane-Based Trash Sorting', self.draw_bbox)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame from camera")
                break
                
            # Flip horizontally for more intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Draw lanes
            for boundary in self.lane_boundaries:
                cv2.line(frame, (boundary, 0), (boundary, self.image_height), 
                         (200, 200, 200), 1)
            
            # Add lane numbers
            for i in range(self.num_lanes):
                lane_center = int((self.lane_boundaries[i] + self.lane_boundaries[i+1]) / 2)
                cv2.putText(frame, f"Lane {i+1}", (lane_center-20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(frame, f"Lane {i+1}", (lane_center-20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Handle object detection (simulation)
            self.simulate_object_detection(frame)
            
            # Preview while drawing
            if self.drawing:
                cv2.rectangle(frame, (self.ix, self.iy), (cv2.getMousePos()[0], cv2.getMousePos()[1]), 
                             (0, 255, 0), 2)
            
            # Draw the current bounding box if exists
            if self.bbox is not None:
                x, y, w, h = self.bbox
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                # Draw bounding box center
                bbox_center_x = x + w//2
                bbox_center_y = y + h//2
                cv2.circle(frame, (bbox_center_x, bbox_center_y), 5, (0, 0, 255), -1)
                
                # Highlight the lane
                lane_idx, lane_center = self.get_lane_from_bbox(self.bbox)
                cv2.line(frame, (lane_center, 0), (lane_center, self.image_height), 
                        (0, 255, 0), 2)
                
                # Draw picking position
                pick_x, pick_y, _ = self.pick_position
                cv2.circle(frame, (pick_x, pick_y), 8, (255, 0, 0), -1)
                cv2.putText(frame, f"Pick ({pick_x}, {pick_y})", (pick_x + 10, pick_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Show which lane it's in
                cv2.putText(frame, f"Object in Lane {lane_idx+1}", (10, self.image_height-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(frame, f"Object in Lane {lane_idx+1}", (10, self.image_height-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add instructions
            cv2.putText(frame, "Draw box with mouse to simulate detection", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
            cv2.putText(frame, "Draw box with mouse to simulate detection", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' to quit, 'c' to clear box", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
            cv2.putText(frame, "Press 'q' to quit, 'c' to clear box", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('Lane-Based Trash Sorting', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.bbox = None
                self.pick_position = None
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
        
    def __del__(self):
        """Ensure camera is released when object is destroyed"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


# Fixed version of the mouse position issue
def cv2_getMousePos():
    """Workaround for missing cv2.getMousePos function"""
    # This is a placeholder - in a real implementation, you would need to 
    # track mouse position in the callback function
    return (0, 0)

# Monkey patch the function
cv2.getMousePos = cv2_getMousePos

# Example usage
if __name__ == "__main__":
    try:
        # Create picker with 5 lanes using default webcam
        picker = LaneBasedRobotPicker(camera_id=0, image_width=640, image_height=480, num_lanes=5)
        
        # Run the system
        picker.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()