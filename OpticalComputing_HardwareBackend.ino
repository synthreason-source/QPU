/*
 * OpticalComputing_HardwareBackend.ino
 * ===================================
 * Arduino firmware companion for optical_binary.py.
 *
 * Drives each light valve from its own Arduino digital output pin
 * (no shift registers). Simple and direct, but the valve count is
 * now hard-capped at however many GPIO pins you list in VALVE_PINS[]
 * -- typically ~10-20 on an Uno/Nano/Mega depending on what else you
 * need. If you outgrow your available pins, you'd need to go back to
 * a shift-register chain (or an I/O expander) instead.
 *
 * Valve state convention: bit = 1 -> valve OPEN, bit = 0 -> valve
 * CLOSED. Invert here (or in your driver stage) if yours is active-low.
 *
 * Framing protocol (must match optical_binary.py):
 *   Host -> MCU:  "FRAME:<N>\n"   ASCII header, N = number of payload bytes
 *                 <N raw bytes>    packed bit data, MSB-first per byte
 *   MCU  -> Host: "ACK\n"          once every valve pin has been set
 *
 * The header is read as a text line (safe -- pure ASCII, always ends
 * in exactly one '\n'). The payload that follows is read as a fixed
 * byte count, NOT scanned for '\n', so binary data can contain any
 * byte value without breaking framing.
 *
 * Only the first NUM_VALVES bits of the incoming payload are applied
 * (one bit per pin, MSB-first per byte, in VALVE_PINS[] order). Any
 * extra bits beyond NUM_VALVES are still read off the wire (to stay
 * in sync with the host) but ignored.
 */

// --- One pin per valve. Edit this list to match your wiring. ---
const int VALVE_PINS[] = {2, 3, 4, 5, 6, 7, 8, 9};
const int NUM_VALVES = sizeof(VALVE_PINS) / sizeof(VALVE_PINS[0]);
const int NUM_PAYLOAD_BYTES = (NUM_VALVES + 7) / 8; // ceil(NUM_VALVES / 8)

const int SENSOR_PIN = A0;  // Analog pin connected to the optical sensor/photodiode
const long BAUD_RATE = 115200;

const unsigned long BYTE_TIMEOUT_MS = 2000; // per-byte read timeout while receiving a frame

// Small fixed buffer, bounded by NUM_VALVES (not by frame size), so
// we can apply all valve pins together for an atomic-looking update.
uint8_t frameBuffer[(NUM_VALVES + 7) / 8];

void setup() {
  Serial.begin(BAUD_RATE);
  Serial.setTimeout(BYTE_TIMEOUT_MS);
  while (!Serial); // Wait for serial port to connect (native USB boards)

  pinMode(SENSOR_PIN, INPUT);

  // Every valve starts CLOSED -- known, safe state before anything
  // else happens.
  for (int i = 0; i < NUM_VALVES; i++) {
    pinMode(VALVE_PINS[i], OUTPUT);
    digitalWrite(VALVE_PINS[i], LOW);
  }
}

// Reads exactly `count` raw bytes off the wire. The first
// NUM_PAYLOAD_BYTES of them are kept in frameBuffer; anything beyond
// that is still read (to stay in sync with the host) but discarded.
// Returns true on success, false on timeout/short read.
bool receiveFrame(long count) {
  for (long i = 0; i < count; i++) {
    unsigned long waitStart = millis();
    while (Serial.available() == 0) {
      if (millis() - waitStart > BYTE_TIMEOUT_MS) {
        return false;
      }
    }
    int b = Serial.read();
    if (b < 0) return false;

    if (i < NUM_PAYLOAD_BYTES) {
      frameBuffer[i] = (uint8_t)b;
    }
    // else: extra byte beyond what we can use -- drained, not stored
  }
  return true;
}

// Applies frameBuffer to the actual valve pins: one bit per valve,
// MSB-first per byte, in VALVE_PINS[] order. This is the moment the
// physical valves change state.
void applyValvesFromBuffer() {
  for (int v = 0; v < NUM_VALVES; v++) {
    int byteIdx = v / 8;
    int bitIdx  = 7 - (v % 8); // MSB-first, matches np.packbits(bitorder="big")
    bool open = (frameBuffer[byteIdx] >> bitIdx) & 0x01;
    digitalWrite(VALVE_PINS[v], open ? HIGH : LOW);
  }
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
        bool ok = receiveFrame(payloadLen);
        if (!ok) {
          Serial.println("ERR:TIMEOUT");
        } else {
          applyValvesFromBuffer(); // this is the actual valve actuation moment
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
