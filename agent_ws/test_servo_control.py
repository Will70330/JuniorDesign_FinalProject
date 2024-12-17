import serial
import time

# Configure the serial port
arduino_port = "COM7"
baud_rate = 9600
timeout = 5.0

try:
    # Establish serial connection to Arduino
    arduino = serial.Serial(port=arduino_port, baudrate=baud_rate, timeout=timeout)
    time.sleep(2) # Wait for arduino to initialize
    print("Connected to Arduino. Enter servo positions (0-180). 'q' to quit.")

    while True:
        user_input = input("Enter Servo Position: ").strip()

        # Exit condition
        if user_input.lower() == "q":
            print("Exiting...")
            break

        # Send data if input is valid number
        if user_input.isdigit():
            position = int(user_input)
            if 0 <= position <= 180:
                arduino.write(f"{position}\n".encode())
                print(f"Sent Position: {position}")
            else:
                print("Error: Invalid Position provided (range is 0 to 180).")
        else:
            print("Invalid input, enter a number or 'q' to exit.")

except serial.SerialException as e:
    print(f"Error: Could not open serial port {arduino_port}: {e}")
finally:
    if 'arduino' in locals() and arduino.is_open:
        arduino.close()
        print("Serial connection closed.")

