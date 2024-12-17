#include <Servo.h>

Servo myServo;          // Servo Object
const int servoPin = 9; // Data Pin Servo is connected to
int servoPos = 0;       // Variable to store the servo position


void setup() {
  Serial.begin(9600);
  myServo.attach(servoPin);
  Serial.println("Servo Control Ready. Send position values (0-180):");
}

void loop() {
  // Check for available data on port
  if (Serial.available() > 0){
    String input = Serial.readStringUntil('\n');  // Read until newline character
    input.trim();                                 // Remove any whitespaces or newline characters
    int value = input.toInt();                    // Convert the input to an integer

    if (value >= 0 && value <= 180) {
      servoPos = value;         // Update Servo Position
      myServo.write(servoPos);  // Move the servo
      Serial.print("Servo moved to: ");
      Serial.println(servoPos);
    } else {
      Serial.println("Invalid input, please only send values between 0 and 180.");
    }
  }
}
