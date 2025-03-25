import serial
import time

class Arduino:
    def __init__(self, port="COM3", baud_rate=9600, timeout=1):
        """
        Initialize the serial connection to the Arduino.
        
        :param port: Serial port name (e.g., 'COM9' on Windows, '/dev/ttyUSB0' on Linux)
        :param baud_rate: Communication speed (must match Arduino's Serial.begin() value)
        :param timeout: Connection timeout in seconds
        """
        try:
            self.ser = serial.Serial(
                port=port, 
                baudrate=baud_rate, 
                timeout=timeout
            )
            # Give the Arduino a moment to reset
            time.sleep(2)
            print(f"Connected to Arduino on {port}")
        except serial.SerialException as e:
            print(f"Error connecting to Arduino: {e}")
            raise
    
    def send(self, command: str):
        """
        Send a command to the Arduino.
        
        :param command: Command string to send (e.g., 'S1A90S2A45')
        """
        try:
            # Encode the command and add a newline (which triggers command processing)
            full_command = command + '\n'
            self.ser.write(full_command.encode('utf-8'))
            
            # Optional: Read and print Arduino's response
            response = self.ser.readline().decode('utf-8').strip()
            if response:
                print(f"Arduino response: {response}")
        except Exception as e:
            print(f"Error sending command: {e}")
    
    def close(self):
        """
        Close the serial connection.
        """
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
            print("Serial connection closed")

# Example usage
if __name__ == "__main__":
    try:
        # Create Arduino connection
        arduino = Arduino()
        
        # Example commands
        # arduino.send("S1A70")  # Move servo 1 to 90 degrees
        # time.sleep(1)
        arduino.send("S1A50S3A39S5A130S9A70")  # Move servo 2 to 45 degrees and servo 3 to 180 degrees
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Ensure connection is closed
        if 'arduino' in locals():
            arduino.close()