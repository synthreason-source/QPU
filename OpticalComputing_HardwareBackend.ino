/*
 * OpticalComputing_HardwareBackend.ino
 * ===================================
 * Arduino firmware companion for optical_binary.py.
 *
 * Drives an arbitrary number of light valves by daisy-chaining
 * 74HC595 (or compatible) shift registers: each additional register
 * adds 8 more valve outputs, chained off the same 3 control pins
 * (DATA / CLOCK / LATCH). Valve count is therefore limited only by
 * how many registers you physically chain -- not by GPIO count.
 *
 * Framing protocol (must match optical_binary.py):
 *   Host -> MCU:  "FRAME:<N>\n"   ASCII header, N = number of payload bytes
 *                 <N raw bytes>    packed bit data, MSB-first per byte
 *   MCU  -> Host: "ACK\n"          once every byte is shifted out and latched
 *
 * The header is read as a text line (safe, since it's pure ASCII and
 * always ends in exactly one '\n'). The payload that follows is read
 * as a fixed number of raw bytes, NOT scanned for '\n' -- so binary
 * data can contain any byte value, including 0x0A, without breaking
 * framing. Each payload byte is shifted straight into the register
 * chain as it arrives (no full-frame buffering), so the number of
 * valves you can address isn't capped by the MCU's RAM either.
 */

const int DATA_PIN  = 11; // to 74HC595 pin 14 (DS)
const int CLOCK_PIN = 12; // to 74HC595 pin 11 (SHCP)
const int LATCH_PIN = 10; // to 74HC595 pin 12 (STCP)

const int SENSOR_PIN = A0;  // Analog pin connected to the optical sensor/photodiode
const long BAUD_RATE = 115200;

const unsigned long BYTE_TIMEOUT_MS = 2000; // per-byte read timeout while receiving a frame

void setup() {
  Serial.begin(BAUD_RATE);
  Serial.setTimeout(BYTE_TIMEOUT_MS);
  while (!Serial); // Wait for serial port to connect (native USB boards)

  pinMode(DATA_PIN, OUTPUT);
  pinMode(CLOCK_PIN, OUTPUT);
  pinMode(LATCH_PIN, OUTPUT);
  pinMode(SENSOR_PIN, INPUT);

  digitalWrite(LATCH_PIN, LOW);
}

// Reads exactly `count` raw bytes one at a time, shifting each one
// into the register chain as soon as it arrives (no full-frame
// buffer, so this scales past the MCU's available RAM).
// Returns true on success, false on timeout/short read.
bool streamFrameIntoShiftRegisters(long count) {
  for (long i = 0; i < count; i++) {
    while (Serial.available() == 0) {
      // Bail out if the host stalls mid-frame
      static unsigned long waitStart = 0;
      if (waitStart == 0) waitStart = millis();
      if (millis() - waitStart > BYTE_TIMEOUT_MS) {
        waitStart = 0;
        return false;
      }
    }
    int b = Serial.read();
    if (b < 0) return false;

    // Shift this byte through the whole daisy-chain. Nothing is
    // latched (i.e. nothing physically changes on the valves) until
    // every byte has been shifted in and latchAll() is called --
    // this keeps the exposure atomic across the whole matrix.
    shiftOut(DATA_PIN, CLOCK_PIN, MSBFIRST, (uint8_t)b);
  }
  return true;
}

void latchAll() {
  digitalWrite(LATCH_PIN, LOW);
  digitalWrite(LATCH_PIN, HIGH);
  digitalWrite(LATCH_PIN, LOW);
}

void loop() {
  if (Serial.available() > 0) {
    // Header format: "FRAME:<N>\n" -- safe to read as a line since it's
    // plain ASCII and always terminates in exactly one newline. Only
    // the payload that follows is binary, and that's read as an exact
    // byte count below, never scanned for '\n'.
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command.startsWith("FRAME:")) {
      int colonIdx = command.indexOf(':');
      long payloadLen = command.substring(colonIdx + 1).toInt();
      if (payloadLen <= 0) {
        Serial.println("ERR:BAD_LENGTH");
      } else {
        bool ok = streamFrameIntoShiftRegisters(payloadLen);
        if (!ok) {
          Serial.println("ERR:TIMEOUT");
        } else {
          latchAll(); // apply the whole matrix to the valves atomically
          Serial.println("ACK");
        }
      }
    }
    else if (command == "TRIGGER_EXPOSURE") {
      // Read optical intensity from the physical sensor
      int sensorValue = analogRead(SENSOR_PIN);
      float intensity = (float)sensorValue / 1023.0 * 5.0;

      Serial.print("INTENSITY:");
      Serial.println(intensity, 4);
    }
  }
}
