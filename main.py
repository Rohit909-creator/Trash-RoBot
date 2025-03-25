from VideoAPI import CircularRegionRobotPicker
from SerialCom import Arduino
import threading
import time

picker = None

def bot():
    global picker
    arduino = Arduino()
    try:
        while True:
            # Safely get pickinfo, with error handling
            if picker and hasattr(picker, 'pickinfo') and picker.pickinfo:
                try:
                    region_index = picker.pickinfo.get("region_index")
                    
                    if region_index is not None:
                        print(f"Detected region: {region_index}")
                        
                        # Universal servo sequence for now
                        servo_sequence = "S1A70S3A39S5A100S9A70"
                        arduino.send(servo_sequence)
                        
                        # Small delay to prevent tight looping
                        time.sleep(1)
                
                except Exception as e:
                    print(f"Error processing pickinfo: {e}")
                    time.sleep(1)
            else:
                # If pickinfo is not available, wait a bit
                time.sleep(0.5)
    
    except Exception as e:
        print(f"Error in robot thread: {e}")
    finally:
        arduino.close()

# Initialize picker
picker = CircularRegionRobotPicker(0)
print("Initial pickinfo:", picker.pickinfo)

# Start robot thread
robot = threading.Thread(target=bot, daemon=True)
robot.start()

# Run picker
picker.run()