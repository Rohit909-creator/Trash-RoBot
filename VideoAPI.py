import cv2
import numpy as np
import time

# S1A24 S3A39 S5A152 S9A70

# Gripper
# Open S7A81
# Close S7A1

class CircularRegionRobotPicker:
    def __init__(self, camera_id=1, image_width=640, image_height=480, num_regions=5):
        """
        Initialize the circle-region-based picker system with live camera feed
        
        Args:
            camera_id: Camera device ID (default 0 for primary webcam)
            image_width: Width of the camera frame in pixels
            image_height: Height of the camera frame in pixels
            num_regions: Number of circular regions to create
        """
        self.image_width = image_width
        self.image_height = image_height
        self.num_regions = num_regions
        self.pickinfo = None
        
        # Create circular regions with hardcoded positions and robotic arm angles
        self.circular_regions = []
        
        # Calculate positions for the circles in a row across the image
        spacing = self.image_width // (num_regions + 1)
        y_position = self.image_height // 2
        
        # Define regions with (center_x, center_y, radius, [angle1, angle2, angle3])
        # The angles represent the robot arm joint angles for picking from this region
        
        locations = [(475, 100), (550, 380), (518, 240), (151, 101), (307, 100)]
        for i in range(num_regions):
            center_x = spacing * (i + 1)
            # Hardcoded angles for each region (simulated joint angles in degrees)
            # Format: [base_angle, shoulder_angle, elbow_angle]
            angles = [
                -45 + (90 / (num_regions-1)) * i,  # Base rotates from -45 to +45 degrees
                90,                               # Shoulder angle
                45                                # Elbow angle
            ]
            # self.circular_regions.append({
            #     'center': (center_x, y_position),
            #     'radius': 50,
            #     'angles': angles,
            #     'index': i
            # })
            
            self.circular_regions.append({
                'center': locations[i],
                'radius': 50,
                'angles': angles,
                'index': i
            })

        angles = [
            -45 + (90 / (num_regions-1)) * i,  # Base rotates from -45 to +45 degrees
            90,                               # Shoulder angle
            45                                # Elbow angle
        ]
        # self.circular_regions.append({
        #     'center': (200, 300),
        #     'radius': 50,
        #     'angles': angles,
        #     'index': 0
        # })
        
        # self.circular_regions.append({
        #     'center': (448, 32),
        #     'radius': 50,
        #     'angles': angles,
        #     'index': 1
        # })
        
        # self.circular_regions.append({
        #     'center': (448, 32),
        #     'radius': 50,
        #     'angles': angles,
        #     'index': 2
        # })
        
        # self.circular_regions.append({
        #     'center': (448, 32),
        #     'radius': 50,
        #     'angles': angles,
        #     'index': 3
        # })
        
        # self.circular_regions.append({
        #     'center': (448, 32),
        #     'radius': 50,
        #     'angles': angles,
        #     'index': 4
        # })
            
        
        print(f"Created {num_regions} circular regions with robot arm angles")
        for i, region in enumerate(self.circular_regions):
            print(f"Region {i}: center={region['center']}, angles={region['angles']}")
        
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
        self.active_region = None
        self.mouse_x, self.mouse_y = 0, 0  # Track mouse position for drawing
        
    def check_bbox_in_region(self, bbox):
        """
        Check if a bounding box overlaps with any circular region
        
        Args:
            bbox: Bounding box in format [x_min, y_min, width, height]
            
        Returns:
            region: The region dictionary if overlap found, None otherwise
        """
        # Calculate center of the bounding box
        bbox_center_x = bbox[0] + bbox[2]//2
        bbox_center_y = bbox[1] + bbox[3]//2
        
        # Check if this point is inside any of our circular regions
        for region in self.circular_regions:
            # Calculate distance from bbox center to circle center
            center_x, center_y = region['center']
            distance = np.sqrt((bbox_center_x - center_x)**2 + (bbox_center_y - center_y)**2)
            
            # If distance is less than radius, bbox center is inside circle
            if distance < region['radius']:
                return region
                
        # No overlap found
        return None
    
    def calculate_pick_position(self, bbox):
        """
        Calculate the position and angles to move the robot arm for picking
        
        Args:
            bbox: Bounding box in format [x_min, y_min, width, height]
            
        Returns:
            pick_info: Dictionary with position and angles for picking
        """
        region = self.check_bbox_in_region(bbox)
        
        if region:
            # Use the region's hardcoded angles and center position
            return {
                'position': region['center'],
                'angles': region['angles'],
                'region_index': region['index']
            }
        else:
            # Not in any region - return center of bbox with default angles
            center_x = bbox[0] + bbox[2]//2
            center_y = bbox[1] + bbox[3]//2
            return {
                'position': (center_x, center_y),
                'angles': [0, 90, 45],  # Default angles
                'region_index': None
            }
    
    def draw_bbox(self, event, x, y, flags, param):
        """Mouse callback function for drawing bounding boxes"""
        # Update current mouse position for drawing preview
        self.mouse_x, self.mouse_y = x, y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            
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
            pick_info = self.calculate_pick_position(self.bbox)
            self.active_region = self.check_bbox_in_region(self.bbox)
            
            print(f"Bbox: {self.bbox}")
            print(f"Pick position: {pick_info['position']}")
            print(f"Robot angles: {pick_info['angles']}")
            if pick_info['region_index'] is not None:
                print(f"In region: {pick_info['region_index']}")
                self.pickinfo = pick_info
            else:
                print("Not in any region")
                self.pickinfo = None
    
    def simulate_object_detection(self, frame):
        """
        Allow manual drawing of bounding boxes to simulate object detection
        or connect to an actual object detection system
        """
        # In a production system, you would replace this with your object
        # detection logic and just return the detected bounding boxes
        
        # For simulation, we use mouse drawing - this is handled in the draw_bbox callback
        pass
    
    def run(self):
        """Run the circular region picker with live camera feed"""
        cv2.namedWindow('Circular Region Picker')
        cv2.setMouseCallback('Circular Region Picker', self.draw_bbox)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame from camera")
                break
                
            # Flip horizontally for more intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Draw circular regions
            for i, region in enumerate(self.circular_regions):
                center = region['center']
                radius = region['radius']
                angles = region['angles']
                
                # Draw circle with index
                cv2.circle(frame, center, radius, (0, 255, 0), 2)
                cv2.putText(frame, f"Region {i+1}", (center[0]-30, center[1]-radius-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(frame, f"Region {i+1}", (center[0]-30, center[1]-radius-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Draw angles info
                angle_text = f"Angles: {angles[0]:.0f}°, {angles[1]:.0f}°, {angles[2]:.0f}°"
                cv2.putText(frame, angle_text, (center[0]-radius, center[1]+radius+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                cv2.putText(frame, angle_text, (center[0]-radius, center[1]+radius+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Preview while drawing
            if self.drawing:
                cv2.rectangle(frame, (self.ix, self.iy), (self.mouse_x, self.mouse_y), 
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
                
                # Get pick information
                pick_info = self.calculate_pick_position(self.bbox)
                pick_x, pick_y = pick_info['position']
                
                # Draw picking position
                cv2.circle(frame, (pick_x, pick_y), 8, (255, 0, 0), -1)
                cv2.putText(frame, f"Pick ({pick_x}, {pick_y})", (pick_x + 10, pick_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Show region and angles if in a region
                if self.active_region is not None:
                    region_idx = self.active_region['index']
                    angles = self.active_region['angles']
                    
                    cv2.putText(frame, f"Object in Region {region_idx+1}", (10, self.image_height-50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                    cv2.putText(frame, f"Object in Region {region_idx+1}", (10, self.image_height-50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    angle_text = f"Robot angles: {angles[0]:.0f}°, {angles[1]:.0f}°, {angles[2]:.0f}°"
                    cv2.putText(frame, angle_text, (10, self.image_height-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                    cv2.putText(frame, angle_text, (10, self.image_height-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, "Object not in any region", (10, self.image_height-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add instructions
            cv2.putText(frame, "Draw box with mouse to simulate detection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
            cv2.putText(frame, "Draw box with mouse to simulate detection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' to quit, 'c' to clear box", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
            cv2.putText(frame, "Press 'q' to quit, 'c' to clear box", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('Circular Region Picker', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.bbox = None
                self.active_region = None
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
        
    def __del__(self):
        """Ensure camera is released when object is destroyed"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

# Example usage
if __name__ == "__main__":
    try:
        # Create picker with 5 circular regions using default webcam
        picker = CircularRegionRobotPicker(camera_id=0, image_width=640, image_height=480, num_regions=5)
        
        # Run the system
        picker.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()