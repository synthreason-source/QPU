/*
 * OpticalComputing_HardwareBackend.ino
 * ===================================
 * Arduino firmware companion for optical_binary.py.
 * Handles light valve matrix framing over USB/Serial and captures
 * simulated or analog optical sensor readings.
 */

const int SENSOR_PIN = A0;  // Analog pin connected to the optical sensor/photodiode
const long BAUD_RATE = 115200;

void setup() {
  Serial.begin(BAUD_RATE);
  while (!Serial); // Wait for serial port to connect (native USB boards)
  
  pinMode(SENSOR_PIN, INPUT);
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    // Check for incoming frame configuration for light valves
    if (command == "FRAME_START") {
      // Read until FRAME_END marker
      // Note: In a real implementation, read the packed bit stream bytes 
      // to drive shift registers or multiplexers connected to your light valves.
      
      bool readingFrame = true;
      while (readingFrame && Serial.available()) {
        String line = Serial.readStringUntil('\n');
        if (line == "FRAME_END") {
          readingFrame = false;
        }
      }
      
      // Acknowledge successful receipt and application of the mask
      Serial.println("ACK");
    }
    
    // Check for sensor exposure trigger command
    else if (command == "TRIGGER_EXPOSURE") {
      // Read optical intensity from the physical sensor
      // (Using analog read as a stand-in for DC field component intensity)
      int sensorValue = analogRead(SENSOR_PIN);
      
      // Map or scale value to a floating point intensity representation
      float intensity = (float)sensorValue / 1023.0 * 5.0; 

      // Send response matching Python script expectation: "INTENSITY:<value>"
      Serial.print("INTENSITY:");
      Serial.println(intensity, 4);
    }
  }
}
