#include <Arduino.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

// ============================================================
// Arduino Uno Optical Processor
// Two independently controlled light valves
// ============================================================
//
// 74HC4051:
//
//   S0 -> D7
//   S1 -> D8
//   S2 -> D9
//   EN -> D10
//   Z  -> A0
//
// 74HC4051 channels:
//
//   Y0 -> operand A photodiode
//   Y1 -> operand B photodiode
//   Y2 -> result photodiode
//   Y3 -> ambient photodiode
//
// Light-valve driver:
//
//   D5 -> Valve A PWM
//   D6 -> Valve B PWM
//   D4 -> common valve-driver enable
//
// IMPORTANT:
//
//   Use MOSFETs or dedicated valve drivers.
//   Do not connect valves directly to Arduino pins.
//
// Serial speed:
//
//   115200 baud
//
// Commands:
//
//   help
//   status
//   read
//   calibrate
//   ambient
//   threshold 500
//   hysteresis 20
//   samples 8
//
//   operation AND
//   operation OR
//   operation XOR
//   operation NAND
//   operation NOR
//   operation XNOR
//   operation IMPLIES
//   operation NIMPLY
//   operation NOT
//   operation HALFADD
//   operation ADD32
//   operation SHR 3
//   operation SHL 3
//   operation ROTR 7
//
//   rega 0x12345678
//   regb 0x87654321
//   execute
//
//   valvea 255
//   valveb 255
//   valves 255 128
//   stop
//
//   test
//   sha256
//   sha256 abc
//   sha256 hello world
//
//   trace on
//   trace off
//   stream on
//   stream off
//
// ============================================================


// ============================================================
// Pin configuration
// ============================================================

const uint8_t ADC_PIN = A0;

// 74HC4051 address pins
const uint8_t MUX_S0 = 7;
const uint8_t MUX_S1 = 8;
const uint8_t MUX_S2 = 9;
const uint8_t MUX_ENABLE = 10;

// Two independent valve PWM outputs
const uint8_t VALVE_A_PWM_PIN = 5;
const uint8_t VALVE_B_PWM_PIN = 6;

// Common enable for both valve drivers
const uint8_t VALVES_ENABLE_PIN = 4;


// ============================================================
// Multiplexer channels
// ============================================================

const uint8_t MUX_OPERAND_A = 0;
const uint8_t MUX_OPERAND_B = 1;
const uint8_t MUX_RESULT = 2;
const uint8_t MUX_AMBIENT = 3;


// ============================================================
// Constants
// ============================================================

const uint16_t ADC_MAX_VALUE = 1023;

const uint16_t DEFAULT_THRESHOLD = 500;
const uint16_t DEFAULT_HYSTERESIS = 20;

const uint8_t DEFAULT_VALVE_A_LEVEL = 255;
const uint8_t DEFAULT_VALVE_B_LEVEL = 255;
const uint8_t DEFAULT_SAMPLE_COUNT = 8;

const uint16_t MUX_SETTLE_TIME_US = 150;
const uint16_t VALVE_SETTLE_TIME_US = 300;

const uint32_t MAX_OPTICAL_EVENTS = 1000000UL;

const uint8_t COMMAND_BUFFER_SIZE = 160;

const uint8_t SHA256_BLOCK_SIZE = 64;
const uint8_t SHA256_DIGEST_SIZE = 32;


// ============================================================
// Result structures
// ============================================================

struct HalfAdderResult {
  bool sum;
  bool carry;
};

struct FullAdderResult {
  bool sum;
  bool carry;
};


// ============================================================
// Operation identifiers
// ============================================================

enum OperationCode {
  OP_AND,
  OP_OR,
  OP_XOR,
  OP_NAND,
  OP_NOR,
  OP_XNOR,
  OP_IMPLIES,
  OP_NIMPLY,
  OP_NOT,
  OP_HALFADD,
  OP_ADD32,
  OP_SHR,
  OP_SHL,
  OP_ROTR
};


// ============================================================
// SHA-256 constants
// ============================================================

const uint32_t SHA256_K[64] = {
  0x428A2F98UL, 0x71374491UL,
  0xB5C0FBCFUL, 0xE9B5DBA5UL,
  0x3956C25BUL, 0x59F111F1UL,
  0x923F82A4UL, 0xAB1C5ED5UL,
  0xD807AA98UL, 0x12835B01UL,
  0x243185BEUL, 0x550C7DC3UL,
  0x72BE5D74UL, 0x80DEB1FEUL,
  0x9BDC06A7UL, 0xC19BF174UL,
  0xE49B69C1UL, 0xEFBE4786UL,
  0x0FC19DC6UL, 0x240CA1CCUL,
  0x2DE92C6FUL, 0x4A7484AAUL,
  0x5CB0A9DCUL, 0x76F988DAUL,
  0x983E5152UL, 0xA831C66DUL,
  0xB00327C8UL, 0xBF597FC7UL,
  0xC6E00BF3UL, 0xD5A79147UL,
  0x06CA6351UL, 0x14292967UL,
  0x27B70A85UL, 0x4D2C6DFCUL,
  0x53380D13UL, 0x650A7354UL,
  0x766A0ABBUL, 0x81C2C92EUL,
  0x92722C85UL, 0xA2BFE8A1UL,
  0xA81A664BUL, 0xC24B8B70UL,
  0xC76C51A3UL, 0xD192E819UL,
  0xD6990624UL, 0x106AA070UL,
  0x19A4C116UL, 0x1E376C08UL,
  0x2748774CUL, 0x34B0BCB5UL,
  0x391C0CB3UL, 0x4ED8AA4AUL,
  0x5B9CCA4FUL, 0x682E6FF3UL,
  0x748F82EEUL, 0x78A5636FUL,
  0x84C87814UL, 0x8CC70208UL,
  0x90BEFFFAUL, 0xA4506CEBUL,
  0xBEF9A3F7UL, 0xC67178F2UL
};

const uint32_t SHA256_H0[8] = {
  0x6A09E667UL,
  0xBB67AE85UL,
  0x3C6EF372UL,
  0xA54FF53AUL,
  0x510E527FUL,
  0x9B05688CUL,
  0x1F83D9ABUL,
  0x5BE0CD19UL
};


// ============================================================
// Global state
// ============================================================

OperationCode currentOperation = OP_AND;

char currentOperationName[16] = "AND";

uint8_t operationAmount = 0;

uint16_t photodiodeThreshold = DEFAULT_THRESHOLD;
uint16_t photodiodeHysteresis = DEFAULT_HYSTERESIS;

uint8_t sampleCount = DEFAULT_SAMPLE_COUNT;

uint16_t ambientLevel = 0;
uint16_t operandALevel = 0;
uint16_t operandBLevel = 0;
uint16_t resultLevel = 0;

bool operandABit = false;
bool operandBBit = false;
bool resultBit = false;

uint8_t valveALevel = DEFAULT_VALVE_A_LEVEL;
uint8_t valveBLevel = DEFAULT_VALVE_B_LEVEL;

bool valvesEnabled = false;

bool traceEnabled = false;
bool streamEnabled = false;

uint32_t opticalEvents = 0;
uint32_t operationCount = 0;
uint32_t errorCount = 0;

uint32_t registerA = 0;
uint32_t registerB = 0;
uint32_t registerResult = 0;
uint32_t registerCarry = 0;

char commandBuffer[COMMAND_BUFFER_SIZE];
uint8_t commandLength = 0;


// ============================================================
// Function prototypes
// ============================================================

void printBit(bool value);
void printHex32(uint32_t value);
void printSeparator();

void uppercaseInPlace(char* text);
char* skipSpaces(char* text);
char* nextToken(char*& text);
uint32_t parseNumber(const char* text);

void muxSelect(uint8_t channel);
uint16_t readMuxRaw(uint8_t channel);
uint16_t readPhotodiode(uint8_t channel);

void valvesOff();
void valvesEnable();
void setValveA(uint8_t level);
void setValveB(uint8_t level);
void setBothValves(uint8_t levelA, uint8_t levelB);
void driveOperandBits(bool a, bool b);
void pulseValveA(uint8_t level, uint16_t durationUs);
void pulseValveB(uint8_t level, uint16_t durationUs);

bool photodiodeToBit(uint16_t level);
void sampleOpticalInputs();
void printOpticalState();

bool opticalAND(bool a, bool b);
bool opticalOR(bool a, bool b);
bool opticalXOR(bool a, bool b);
bool opticalNOT(bool a);
bool opticalNAND(bool a, bool b);
bool opticalNOR(bool a, bool b);
bool opticalXNOR(bool a, bool b);
bool opticalIMPLIES(bool a, bool b);
bool opticalNIMPLY(bool a, bool b);

HalfAdderResult opticalHalfAdder(bool a, bool b);

FullAdderResult opticalFullAdder(
  bool a,
  bool b,
  bool carryIn
);

uint32_t opticalAND32(uint32_t a, uint32_t b);
uint32_t opticalOR32(uint32_t a, uint32_t b);
uint32_t opticalXOR32(uint32_t a, uint32_t b);
uint32_t opticalNOT32(uint32_t value);
uint32_t opticalNAND32(uint32_t a, uint32_t b);
uint32_t opticalNOR32(uint32_t a, uint32_t b);
uint32_t opticalXNOR32(uint32_t a, uint32_t b);
uint32_t opticalIMPLIES32(uint32_t a, uint32_t b);
uint32_t opticalNIMPLY32(uint32_t a, uint32_t b);
uint32_t opticalADD32(uint32_t a, uint32_t b);
uint32_t opticalSHR32(uint32_t value, uint8_t amount);
uint32_t opticalSHL32(uint32_t value, uint8_t amount);
uint32_t opticalROTR32(uint32_t value, uint8_t amount);

bool executeSingleOperation(bool a, bool b);
void executeOpticalOperation();

void calibrateThreshold();
void automaticAmbientCalibration();

bool testBitOperation(
  const char* name,
  bool actual,
  bool expected
);

bool testWordOperation(
  const char* name,
  uint32_t actual,
  uint32_t expected
);

void runSelfTest();

uint32_t sha256Ch(
  uint32_t x,
  uint32_t y,
  uint32_t z
);

uint32_t sha256Maj(
  uint32_t x,
  uint32_t y,
  uint32_t z
);

uint32_t sha256BigSigma0(uint32_t x);
uint32_t sha256BigSigma1(uint32_t x);
uint32_t sha256SmallSigma0(uint32_t x);
uint32_t sha256SmallSigma1(uint32_t x);

uint32_t sha256ReadWord(
  const uint8_t* block,
  uint8_t index
);

void sha256WriteWord(
  uint8_t* output,
  uint8_t index,
  uint32_t value
);

void sha256Hash(
  const uint8_t* message,
  uint16_t messageLength,
  uint8_t digest[SHA256_DIGEST_SIZE]
);

void printSha256Digest(
  const uint8_t digest[SHA256_DIGEST_SIZE]
);

void sha256Command(const char* message);
void runSha256Test();

bool setOperationByName(const char* name);
void printStatus();
void printHelp();

void processCommand(char* command);
void readSerialCommands();
void streamOpticalState();


// ============================================================
// General utilities
// ============================================================

void incrementOpticalEvents()
{
  if (opticalEvents < MAX_OPTICAL_EVENTS) {
    opticalEvents++;
  }
}

void incrementOperationCount()
{
  operationCount++;
}

void printBit(bool value)
{
  Serial.print(value ? 1 : 0);
}

void printHex32(uint32_t value)
{
  char buffer[9];

  snprintf(
    buffer,
    sizeof(buffer),
    "%08lX",
    (unsigned long)value
  );

  Serial.print(buffer);
}

void printSeparator()
{
  Serial.println(F("----------------------------------------"));
}

void uppercaseInPlace(char* text)
{
  while (*text != '\0') {
    *text = toupper(*text);
    text++;
  }
}

char* skipSpaces(char* text)
{
  while (*text == ' ' || *text == '\t') {
    text++;
  }

  return text;
}

char* nextToken(char*& text)
{
  text = skipSpaces(text);

  if (*text == '\0') {
    return nullptr;
  }

  char* start = text;

  while (*text != '\0' &&
         *text != ' ' &&
         *text != '\t') {
    text++;
  }

  if (*text != '\0') {
    *text = '\0';
    text++;
  }

  return start;
}

uint32_t parseNumber(const char* text)
{
  if (text == nullptr) {
    return 0;
  }

  if (text[0] == '0' &&
      (text[1] == 'x' || text[1] == 'X')) {
    return strtoul(text + 2, nullptr, 16);
  }

  return strtoul(text, nullptr, 10);
}


// ============================================================
// 74HC4051 multiplexer
// ============================================================

void muxSelect(uint8_t channel)
{
  channel &= 0x07;

  digitalWrite(
    MUX_S0,
    (channel & 0x01) ? HIGH : LOW
  );

  digitalWrite(
    MUX_S1,
    (channel & 0x02) ? HIGH : LOW
  );

  digitalWrite(
    MUX_S2,
    (channel & 0x04) ? HIGH : LOW
  );

  delayMicroseconds(MUX_SETTLE_TIME_US);
}

uint16_t readMuxRaw(uint8_t channel)
{
  digitalWrite(MUX_ENABLE, LOW);

  muxSelect(channel);

  uint32_t total = 0;

  for (uint8_t i = 0; i < sampleCount; i++) {
    total += analogRead(ADC_PIN);
  }

  return (uint16_t)(total / sampleCount);
}

uint16_t readPhotodiode(uint8_t channel)
{
  return readMuxRaw(channel);
}

bool photodiodeToBit(uint16_t level)
{
  return level >= photodiodeThreshold;
}


// ============================================================
// Two-valve driver
// ============================================================

void valvesOff()
{
  analogWrite(VALVE_A_PWM_PIN, 0);
  analogWrite(VALVE_B_PWM_PIN, 0);

  digitalWrite(VALVES_ENABLE_PIN, LOW);

  valvesEnabled = false;
}

void valvesEnable()
{
  digitalWrite(VALVES_ENABLE_PIN, HIGH);
  valvesEnabled = true;
}

void setValveA(uint8_t level)
{
  valveALevel = level;
  analogWrite(VALVE_A_PWM_PIN, valveALevel);
}

void setValveB(uint8_t level)
{
  valveBLevel = level;
  analogWrite(VALVE_B_PWM_PIN, valveBLevel);
}

void setBothValves(
  uint8_t levelA,
  uint8_t levelB
)
{
  setValveA(levelA);
  setValveB(levelB);
  valvesEnable();
}

void driveOperandBits(bool a, bool b)
{
  uint8_t levelA = a ? valveALevel : 0;
  uint8_t levelB = b ? valveBLevel : 0;

  setBothValves(levelA, levelB);
}

void pulseValveA(
  uint8_t level,
  uint16_t durationUs
)
{
  setValveA(level);
  valvesEnable();

  delayMicroseconds(durationUs);

  setValveA(0);
}

void pulseValveB(
  uint8_t level,
  uint16_t durationUs
)
{
  setValveB(level);
  valvesEnable();

  delayMicroseconds(durationUs);

  setValveB(0);
}


// ============================================================
// Optical input sampling
// ============================================================

void sampleOpticalInputs()
{
  ambientLevel =
    readPhotodiode(MUX_AMBIENT);

  operandALevel =
    readPhotodiode(MUX_OPERAND_A);

  operandBLevel =
    readPhotodiode(MUX_OPERAND_B);

  resultLevel =
    readPhotodiode(MUX_RESULT);

  operandABit =
    photodiodeToBit(operandALevel);

  operandBBit =
    photodiodeToBit(operandBLevel);

  resultBit =
    photodiodeToBit(resultLevel);
}

void printOpticalState()
{
  Serial.print(F("A="));
  Serial.print(operandALevel);
  Serial.print(F(":"));
  printBit(operandABit);

  Serial.print(F(" B="));
  Serial.print(operandBLevel);
  Serial.print(F(":"));
  printBit(operandBBit);

  Serial.print(F(" RESULT="));
  Serial.print(resultLevel);
  Serial.print(F(":"));
  printBit(resultBit);

  Serial.print(F(" AMBIENT="));
  Serial.println(ambientLevel);
}


// ============================================================
// Primitive optical operations (Hardware-in-the-Loop)
// ============================================================

bool opticalAND(bool a, bool b)
{
  incrementOpticalEvents();

  driveOperandBits(a, b);
  delayMicroseconds(VALVE_SETTLE_TIME_US);
  sampleOpticalInputs();
  bool result = resultBit;

  if (traceEnabled) {
    Serial.print(F("AND("));
    printBit(a);
    Serial.print(',');
    printBit(b);
    Serial.print(F(")->Result="));
    printBit(result);
    Serial.println();
  }

  return result;
}

bool opticalOR(bool a, bool b)
{
  incrementOpticalEvents();

  driveOperandBits(a, b);
  delayMicroseconds(VALVE_SETTLE_TIME_US);
  sampleOpticalInputs();
  bool result = resultBit;

  if (traceEnabled) {
    Serial.print(F("OR("));
    printBit(a);
    Serial.print(',');
    printBit(b);
    Serial.print(F(")->Result="));
    printBit(result);
    Serial.println();
  }

  return result;
}

bool opticalXOR(bool a, bool b)
{
  incrementOpticalEvents();

  driveOperandBits(a, b);
  delayMicroseconds(VALVE_SETTLE_TIME_US);
  sampleOpticalInputs();
  bool result = resultBit;

  if (traceEnabled) {
    Serial.print(F("XOR("));
    printBit(a);
    Serial.print(',');
    printBit(b);
    Serial.print(F(")->Result="));
    printBit(result);
    Serial.println();
  }

  return result;
}

bool opticalNOT(bool a)
{
  incrementOpticalEvents();

  // For NOT, we drive operand A with the inverted state or single-valve setup
  driveOperandBits(!a, false);
  delayMicroseconds(VALVE_SETTLE_TIME_US);
  sampleOpticalInputs();
  bool result = resultBit;

  if (traceEnabled) {
    Serial.print(F("NOT("));
    printBit(a);
    Serial.print(F(")->Result="));
    printBit(result);
    Serial.println();
  }

  return result;
}

bool opticalNAND(bool a, bool b)
{
  return opticalNOT(
    opticalAND(a, b)
  );
}

bool opticalNOR(bool a, bool b)
{
  return opticalNOT(
    opticalOR(a, b)
  );
}

bool opticalXNOR(bool a, bool b)
{
  return opticalNOT(
    opticalXOR(a, b)
  );
}

bool opticalIMPLIES(bool a, bool b)
{
  bool notA = opticalNOT(a);
  return opticalOR(notA, b);
}

bool opticalNIMPLY(bool a, bool b)
{
  bool notB = opticalNOT(b);
  return opticalAND(a, notB);
}


// ============================================================
// Adders
// ============================================================

HalfAdderResult opticalHalfAdder(bool a, bool b)
{
  HalfAdderResult result;

  result.sum =
    opticalXOR(a, b);

  result.carry =
    opticalAND(a, b);

  if (traceEnabled) {
    Serial.print(F("HALFADD("));
    printBit(a);
    Serial.print(',');
    printBit(b);
    Serial.print(F(")->SUM="));
    printBit(result.sum);
    Serial.print(F(" CARRY="));
    printBit(result.carry);
    Serial.println();
  }

  return result;
}

FullAdderResult opticalFullAdder(
  bool a,
  bool b,
  bool carryIn
)
{
  HalfAdderResult first;
  HalfAdderResult second;

  FullAdderResult result;

  first =
    opticalHalfAdder(a, b);

  second =
    opticalHalfAdder(first.sum, carryIn);

  result.sum =
    second.sum;

  result.carry =
    opticalOR(first.carry, second.carry);

  if (traceEnabled) {
    Serial.print(F("FULLADD("));
    printBit(a);
    Serial.print(',');
    printBit(b);
    Serial.print(',');
    printBit(carryIn);
    Serial.print(F(")->SUM="));
    printBit(result.sum);
    Serial.print(F(" CARRY="));
    printBit(result.carry);
    Serial.println();
  }

  return result;
}


// ============================================================
// 32-bit operations
// ============================================================

uint32_t opticalAND32(uint32_t a, uint32_t b)
{
  uint32_t result = 0;

  for (uint8_t bit = 0; bit < 32; bit++) {
    bool aBit = (a >> bit) & 1UL;
    bool bBit = (b >> bit) & 1UL;

    if (opticalAND(aBit, bBit)) {
      result |= (1UL << bit);
    }
  }

  return result;
}

uint32_t opticalOR32(uint32_t a, uint32_t b)
{
  uint32_t result = 0;

  for (uint8_t bit = 0; bit < 32; bit++) {
    bool aBit = (a >> bit) & 1UL;
    bool bBit = (b >> bit) & 1UL;

    if (opticalOR(aBit, bBit)) {
      result |= (1UL << bit);
    }
  }

  return result;
}

uint32_t opticalXOR32(uint32_t a, uint32_t b)
{
  uint32_t result = 0;

  for (uint8_t bit = 0; bit < 32; bit++) {
    bool aBit = (a >> bit) & 1UL;
    bool bBit = (b >> bit) & 1UL;

    if (opticalXOR(aBit, bBit)) {
      result |= (1UL << bit);
    }
  }

  return result;
}

uint32_t opticalNOT32(uint32_t value)
{
  uint32_t result = 0;

  for (uint8_t bit = 0; bit < 32; bit++) {
    bool inputBit =
      (value >> bit) & 1UL;

    if (opticalNOT(inputBit)) {
      result |= (1UL << bit);
    }
  }

  return result;
}

uint32_t opticalNAND32(uint32_t a, uint32_t b)
{
  return opticalNOT32(
    opticalAND32(a, b)
  );
}

uint32_t opticalNOR32(uint32_t a, uint32_t b)
{
  return opticalNOT32(
    opticalOR32(a, b)
  );
}

uint32_t opticalXNOR32(uint32_t a, uint32_t b)
{
  return opticalNOT32(
    opticalXOR32(a, b)
  );
}

uint32_t opticalIMPLIES32(uint32_t a, uint32_t b)
{
  return opticalOR32(
    opticalNOT32(a),
    b
  );
}

uint32_t opticalNIMPLY32(uint32_t a, uint32_t b)
{
  return opticalAND32(
    a,
    opticalNOT32(b)
  );
}

uint32_t opticalADD32(uint32_t a, uint32_t b)
{
  uint32_t result = 0;
  bool carry = false;

  for (uint8_t bit = 0; bit < 32; bit++) {
    bool aBit = (a >> bit) & 1UL;
    bool bBit = (b >> bit) & 1UL;

    FullAdderResult fullAdder =
      opticalFullAdder(
        aBit,
        bBit,
        carry
      );

    if (fullAdder.sum) {
      result |= (1UL << bit);
    }

    carry = fullAdder.carry;
  }

  registerCarry =
    carry ? 1UL : 0UL;

  return result;
}

uint32_t opticalSHR32(
  uint32_t value,
  uint8_t amount
)
{
  incrementOpticalEvents();

  if (amount >= 32) {
    return 0;
  }

  return value >> amount;
}

uint32_t opticalSHL32(
  uint32_t value,
  uint8_t amount
)
{
  incrementOpticalEvents();

  if (amount >= 32) {
    return 0;
  }

  return value << amount;
}

uint32_t opticalROTR32(
  uint32_t value,
  uint8_t amount
)
{
  incrementOpticalEvents();

  amount %= 32;

  if (amount == 0) {
    return value;
  }

  return (value >> amount) |
         (value << (32 - amount));
}


// ============================================================
// Execute selected operation
// ============================================================

bool executeSingleOperation(bool a, bool b)
{
  switch (currentOperation) {
    case OP_AND:
      return opticalAND(a, b);

    case OP_OR:
      return opticalOR(a, b);

    case OP_XOR:
      return opticalXOR(a, b);

    case OP_NAND:
      return opticalNAND(a, b);

    case OP_NOR:
      return opticalNOR(a, b);

    case OP_XNOR:
      return opticalXNOR(a, b);

    case OP_IMPLIES:
      return opticalIMPLIES(a, b);

    case OP_NIMPLY:
      return opticalNIMPLY(a, b);

    case OP_NOT:
      return opticalNOT(a);

    default:
      return false;
  }
}

void executeOpticalOperation()
{
  sampleOpticalInputs();
  incrementOperationCount();

  driveOperandBits(
    operandABit,
    operandBBit
  );

  delayMicroseconds(VALVE_SETTLE_TIME_US);

  if (currentOperation == OP_HALFADD) {
    HalfAdderResult result =
      opticalHalfAdder(
        operandABit,
        operandBBit
      );

    Serial.print(F("HALFADD A="));
    printBit(operandABit);

    Serial.print(F(" B="));
    printBit(operandBBit);

    Serial.print(F(" SUM="));
    printBit(result.sum);

    Serial.print(F(" CARRY="));
    printBit(result.carry);

    Serial.println();

    return;
  }

  if (currentOperation == OP_ADD32) {
    registerResult =
      opticalADD32(
        registerA,
        registerB
      );

    Serial.print(F("ADD32 RESULT="));
    printHex32(registerResult);

    Serial.print(F(" CARRY="));
    printHex32(registerCarry);

    Serial.println();

    return;
  }

  if (currentOperation == OP_SHR) {
    registerResult =
      opticalSHR32(
        registerA,
        operationAmount
      );

    Serial.print(F("SHR RESULT="));
    printHex32(registerResult);
    Serial.println();

    return;
  }

  if (currentOperation == OP_SHL) {
    registerResult =
      opticalSHL32(
        registerA,
        operationAmount
      );

    Serial.print(F("SHL RESULT="));
    printHex32(registerResult);
    Serial.println();

    return;
  }

  if (currentOperation == OP_ROTR) {
    registerResult =
      opticalROTR32(
        registerA,
        operationAmount
      );

    Serial.print(F("ROTR RESULT="));
    printHex32(registerResult);
    Serial.println();

    return;
  }

  bool output =
    executeSingleOperation(
      operandABit,
      operandBBit
    );

  Serial.print(currentOperationName);

  Serial.print(F(" A="));
  printBit(operandABit);

  Serial.print(F(" B="));
  printBit(operandBBit);

  Serial.print(F(" OUT="));
  printBit(output);

  Serial.println();
}


// ============================================================
// SHA-256
// ============================================================

uint32_t sha256Ch(
  uint32_t x,
  uint32_t y,
  uint32_t z
)
{
  uint32_t xy =
    opticalAND32(x, y);

  uint32_t notX =
    opticalNOT32(x);

  uint32_t notXz =
    opticalAND32(notX, z);

  return opticalXOR32(xy, notXz);
}

uint32_t sha256Maj(
  uint32_t x,
  uint32_t y,
  uint32_t z
)
{
  uint32_t xy =
    opticalAND32(x, y);

  uint32_t xz =
    opticalAND32(x, z);

  uint32_t yz =
    opticalAND32(y, z);

  uint32_t partial =
    opticalXOR32(xy, xz);

  return opticalXOR32(partial, yz);
}

uint32_t sha256BigSigma0(uint32_t x)
{
  uint32_t r2 =
    opticalROTR32(x, 2);

  uint32_t r13 =
    opticalROTR32(x, 13);

  uint32_t r22 =
    opticalROTR32(x, 22);

  return opticalXOR32(
    opticalXOR32(r2, r13),
    r22
  );
}

uint32_t sha256BigSigma1(uint32_t x)
{
  uint32_t r6 =
    opticalROTR32(x, 6);

  uint32_t r11 =
    opticalROTR32(x, 11);

  uint32_t r25 =
    opticalROTR32(x, 25);

  return opticalXOR32(
    opticalXOR32(r6, r11),
    r25
  );
}

uint32_t sha256SmallSigma0(uint32_t x)
{
  uint32_t r7 =
    opticalROTR32(x, 7);

  uint32_t r18 =
    opticalROTR32(x, 18);

  uint32_t s3 =
    opticalSHR32(x, 3);

  return opticalXOR32(
    opticalXOR32(r7, r18),
    s3
  );
}

uint32_t sha256SmallSigma1(uint32_t x)
{
  uint32_t r17 =
    opticalROTR32(x, 17);

  uint32_t r19 =
    opticalROTR32(x, 19);

  uint32_t s10 =
    opticalSHR32(x, 10);

  return opticalXOR32(
    opticalXOR32(r17, r19),
    s10
  );
}

uint32_t sha256ReadWord(
  const uint8_t* block,
  uint8_t index
)
{
  uint8_t offset =
    index * 4;

  uint32_t word = 0;

  word |=
    ((uint32_t)block[offset + 0]) << 24;

  word |=
    ((uint32_t)block[offset + 1]) << 16;

  word |=
    ((uint32_t)block[offset + 2]) << 8;

  word |=
    ((uint32_t)block[offset + 3]);

  return word;
}

void sha256WriteWord(
  uint8_t* output,
  uint8_t index,
  uint32_t value
)
{
  uint8_t offset =
    index * 4;

  output[offset + 0] =
    (uint8_t)(value >> 24);

  output[offset + 1] =
    (uint8_t)(value >> 16);

  output[offset + 2] =
    (uint8_t)(value >> 8);

  output[offset + 3] =
    (uint8_t)value;
}

void sha256Hash(
  const uint8_t* message,
  uint16_t messageLength,
  uint8_t digest[SHA256_DIGEST_SIZE]
)
{
  uint32_t state[8];

  for (uint8_t i = 0; i < 8; i++) {
    state[i] = SHA256_H0[i];
  }

  uint32_t bitLength =
    ((uint32_t)messageLength) * 8UL;

  uint16_t paddedLength =
    messageLength + 1 + 8;

  while ((paddedLength % 64) != 0) {
    paddedLength++;
  }

  uint16_t blockCount =
    paddedLength / SHA256_BLOCK_SIZE;

  uint8_t block[SHA256_BLOCK_SIZE];

  for (uint16_t blockIndex = 0;
       blockIndex < blockCount;
       blockIndex++) {

    uint32_t blockStart =
      ((uint32_t)blockIndex) *
      SHA256_BLOCK_SIZE;

    for (uint8_t i = 0; i < SHA256_BLOCK_SIZE; i++) {
      uint32_t absoluteIndex =
        blockStart + i;

      if (absoluteIndex < messageLength) {
        block[i] =
          message[absoluteIndex];
      } else if (absoluteIndex == messageLength) {
        block[i] = 0x80;
      } else {
        block[i] = 0;
      }
    }

    if (blockIndex == blockCount - 1) {
      block[56] = 0;
      block[57] = 0;
      block[58] = 0;
      block[59] = 0;

      block[60] =
        (uint8_t)(bitLength >> 24);

      block[61] =
        (uint8_t)(bitLength >> 16);

      block[62] =
        (uint8_t)(bitLength >> 8);

      block[63] =
        (uint8_t)bitLength;
    }

    uint32_t w[64];

    for (uint8_t t = 0; t < 16; t++) {
      w[t] =
        sha256ReadWord(block, t);
    }

    for (uint8_t t = 16; t < 64; t++) {
      uint32_t sigma0 =
        sha256SmallSigma0(w[t - 15]);

      uint32_t sigma1 =
        sha256SmallSigma1(w[t - 2]);

      uint32_t part1 =
        opticalADD32(
          w[t - 16],
          sigma0
        );

      uint32_t part2 =
        opticalADD32(
          part1,
          w[t - 7]
        );

      w[t] =
        opticalADD32(
          part2,
          sigma1
        );
    }

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    for (uint8_t t = 0; t < 64; t++) {
      uint32_t sigma1 =
        sha256BigSigma1(e);

      uint32_t choose =
        sha256Ch(e, f, g);

      uint32_t temp1 =
        opticalADD32(h, sigma1);

      temp1 =
        opticalADD32(temp1, choose);

      temp1 =
        opticalADD32(temp1, SHA256_K[t]);

      temp1 =
        opticalADD32(temp1, w[t]);

      uint32_t sigma0 =
        sha256BigSigma0(a);

      uint32_t majority =
        sha256Maj(a, b, c);

      uint32_t temp2 =
        opticalADD32(sigma0, majority);

      h = g;
      g = f;
      f = e;

      e =
        opticalADD32(d, temp1);

      d = c;
      c = b;
      b = a;

      a =
        opticalADD32(temp1, temp2);
    }

    state[0] =
      opticalADD32(state[0], a);

    state[1] =
      opticalADD32(state[1], b);

    state[2] =
      opticalADD32(state[2], c);

    state[3] =
      opticalADD32(state[3], d);

    state[4] =
      opticalADD32(state[4], e);

    state[5] =
      opticalADD32(state[5], f);

    state[6] =
      opticalADD32(state[6], g);

    state[7] =
      opticalADD32(state[7], h);
  }

  for (uint8_t i = 0; i < 8; i++) {
    sha256WriteWord(
      digest,
      i,
      state[i]
    );
  }
}

void printSha256Digest(
  const uint8_t digest[SHA256_DIGEST_SIZE]
)
{
  for (uint8_t i = 0; i < SHA256_DIGEST_SIZE; i++) {
    if (digest[i] < 0x10) {
      Serial.print('0');
    }

    Serial.print(digest[i], HEX);
  }

  Serial.println();
}

void sha256Command(const char* message)
{
  uint16_t length =
    strlen(message);

  if (length > 119) {
    Serial.println(
      F("SHA-256 input must be 119 bytes or less.")
    );

    return;
  }

  uint8_t digest[SHA256_DIGEST_SIZE];

  opticalEvents = 0;

  Serial.println(F("Computing SHA-256..."));

  Serial.print(F("Input: "));
  Serial.println(message);

  uint32_t startTime =
    millis();

  sha256Hash(
    (const uint8_t*)message,
    length,
    digest
  );

  uint32_t elapsed =
    millis() - startTime;

  Serial.print(F("SHA256: "));
  printSha256Digest(digest);

  Serial.print(F("Time_ms: "));
  Serial.println(elapsed);

  Serial.print(F("Optical events: "));
  Serial.println(opticalEvents);
}

bool sha256CompareDigest(
  const uint8_t actual[SHA256_DIGEST_SIZE],
  const uint8_t expected[SHA256_DIGEST_SIZE]
)
{
  for (uint8_t i = 0; i < SHA256_DIGEST_SIZE; i++) {
    if (actual[i] != expected[i]) {
      return false;
    }
  }

  return true;
}

void runSha256Test()
{
  static const char testMessage[] = "abc";

  static const uint8_t expectedDigest[
    SHA256_DIGEST_SIZE
  ] = {
    0xBA, 0x78, 0x16, 0xBF,
    0x8F, 0x01, 0xCF, 0xEA,
    0x41, 0x41, 0x40, 0xDE,
    0x5D, 0xAE, 0x22, 0x23,
    0xB0, 0x03, 0x61, 0xA3,
    0x96, 0x17, 0x7A, 0x9C,
    0xB4, 0x10, 0xFF, 0x61,
    0xF2, 0x00, 0x15, 0xAD
  };

  uint8_t actualDigest[SHA256_DIGEST_SIZE];

  opticalEvents = 0;

  Serial.println(F("SHA-256 known-answer test"));
  Serial.println(F("Message: abc"));

  uint32_t startTime =
    millis();

  sha256Hash(
    (const uint8_t*)testMessage,
    3,
    actualDigest
  );

  uint32_t elapsed =
    millis() - startTime;

  Serial.print(F("Calculated: "));
  printSha256Digest(actualDigest);

  Serial.print(F("Expected:   "));
  printSha256Digest(expectedDigest);

  bool passed =
    sha256CompareDigest(
      actualDigest,
      expectedDigest
    );

  Serial.print(F("Result: "));
  Serial.println(
    passed ? F("PASS") : F("FAIL")
  );

  Serial.print(F("Time_ms: "));
  Serial.println(elapsed);

  Serial.print(F("Optical events: "));
  Serial.println(opticalEvents);

  if (!passed) {
    errorCount++;
  }
}


// ============================================================
// Calibration
// ============================================================

void calibrateThreshold()
{
  valvesOff();

  Serial.println(
    F("Block both operand photodiodes.")
  );

  Serial.println(
    F("Press any key for dark measurement.")
  );

  while (!Serial.available()) {
    delay(10);
  }

  while (Serial.available()) {
    Serial.read();
  }

  uint16_t darkA =
    readPhotodiode(MUX_OPERAND_A);

  uint16_t darkB =
    readPhotodiode(MUX_OPERAND_B);

  uint16_t darkR =
    readPhotodiode(MUX_RESULT);

  Serial.println(
    F("Illuminate both photodiodes.")
  );

  Serial.println(
    F("Press any key for bright measurement.")
  );

  while (!Serial.available()) {
    delay(10);
  }

  while (Serial.available()) {
    Serial.read();
  }

  uint16_t brightA =
    readPhotodiode(MUX_OPERAND_A);

  uint16_t brightB =
    readPhotodiode(MUX_OPERAND_B);

  uint16_t brightR =
    readPhotodiode(MUX_RESULT);

  uint16_t darkAverage =
    (darkA + darkB + darkR) / 3;

  uint16_t brightAverage =
    (brightA + brightB + brightR) / 3;

  if (brightAverage >= darkAverage) {
    photodiodeThreshold =
      darkAverage +
      ((brightAverage - darkAverage) / 2);
  } else {
    photodiodeThreshold =
      brightAverage +
      ((darkAverage - brightAverage) / 2);
  }

  Serial.print(F("Dark average="));
  Serial.println(darkAverage);

  Serial.print(F("Bright average="));
  Serial.println(brightAverage);

  Serial.print(F("Threshold="));
  Serial.println(photodiodeThreshold);
}

void automaticAmbientCalibration()
{
  valvesOff();

  ambientLevel =
    readPhotodiode(MUX_AMBIENT);

  if (ambientLevel < ADC_MAX_VALUE - 100) {
    photodiodeThreshold =
      ambientLevel + 100;
  }

  Serial.print(F("Ambient="));
  Serial.print(ambientLevel);

  Serial.print(F(" Threshold="));
  Serial.println(photodiodeThreshold);
}


// ============================================================
// Tests
// ============================================================

bool testBitOperation(
  const char* name,
  bool actual,
  bool expected
)
{
  if (actual != expected) {
    Serial.print(F("FAIL "));
    Serial.print(name);

    Serial.print(F(" expected="));
    printBit(expected);

    Serial.print(F(" actual="));
    printBit(actual);

    Serial.println();

    errorCount++;
    return false;
  }

  return true;
}

bool testWordOperation(
  const char* name,
  uint32_t actual,
  uint32_t expected
)
{
  if (actual != expected) {
    Serial.print(F("FAIL "));
    Serial.print(name);

    Serial.print(F(" expected="));
    printHex32(expected);

    Serial.print(F(" actual="));
    printHex32(actual);

    Serial.println();

    errorCount++;
    return false;
  }

  return true;
}

void runSelfTest()
{
  errorCount = 0;

  bool values[4][2] = {
    {false, false},
    {false, true},
    {true, false},
    {true, true}
  };

  Serial.println(
    F("Testing Boolean operations...")
  );

  for (uint8_t i = 0; i < 4; i++) {
    bool a = values[i][0];
    bool b = values[i][1];

    testBitOperation(
      "AND",
      opticalAND(a, b),
      a && b
    );

    testBitOperation(
      "OR",
      opticalOR(a, b),
      a || b
    );

    testBitOperation(
      "XOR",
      opticalXOR(a, b),
      a ^ b
    );

    testBitOperation(
      "NAND",
      opticalNAND(a, b),
      !(a && b)
    );

    testBitOperation(
      "NOR",
      opticalNOR(a, b),
      !(a || b)
    );

    testBitOperation(
      "XNOR",
      opticalXNOR(a, b),
      !(a ^ b)
    );

    testBitOperation(
      "IMPLIES",
      opticalIMPLIES(a, b),
      (!a) || b
    );

    testBitOperation(
      "NIMPLY",
      opticalNIMPLY(a, b),
      a && (!b)
    );
  }

  Serial.println(
    F("Testing half-adders...")
  );

  for (uint8_t i = 0; i < 4; i++) {
    bool a = values[i][0];
    bool b = values[i][1];

    HalfAdderResult result =
      opticalHalfAdder(a, b);

    testBitOperation(
      "HALFADD SUM",
      result.sum,
      a ^ b
    );

    testBitOperation(
      "HALFADD CARRY",
      result.carry,
      a && b
    );
  }

  Serial.println(
    F("Testing full-adders...")
  );

  for (uint8_t a = 0; a <= 1; a++) {
    for (uint8_t b = 0; b <= 1; b++) {
      for (uint8_t carry = 0; carry <= 1; carry++) {
        FullAdderResult result =
          opticalFullAdder(
            a,
            b,
            carry
          );

        uint8_t expected =
          a + b + carry;

        testBitOperation(
          "FULLADD SUM",
          result.sum,
          expected & 1
        );

        testBitOperation(
          "FULLADD CARRY",
          result.carry,
          expected > 1
        );
      }
    }
  }

  const uint32_t a = 0x12345678UL;
  const uint32_t b = 0x87654321UL;

  Serial.println(
    F("Testing 32-bit operations...")
  );

  testWordOperation(
    "AND32",
    opticalAND32(a, b),
    a & b
  );

  testWordOperation(
    "OR32",
    opticalOR32(a, b),
    a | b
  );

  testWordOperation(
    "XOR32",
    opticalXOR32(a, b),
    a ^ b
  );

  testWordOperation(
    "NAND32",
    opticalNAND32(a, b),
    ~(a & b)
  );

  testWordOperation(
    "NOR32",
    opticalNOR32(a, b),
    ~(a | b)
  );

  testWordOperation(
    "XNOR32",
    opticalXNOR32(a, b),
    ~(a ^ b)
  );

  testWordOperation(
    "IMPLIES32",
    opticalIMPLIES32(a, b),
    (~a) | b
  );

  testWordOperation(
    "NIMPLY32",
    opticalNIMPLY32(a, b),
    a & (~b)
  );

  testWordOperation(
    "ADD32",
    opticalADD32(a, b),
    a + b
  );

  testWordOperation(
    "SHR",
    opticalSHR32(a, 3),
    a >> 3
  );

  testWordOperation(
    "SHL",
    opticalSHL32(a, 3),
    a << 3
  );

  testWordOperation(
    "ROTR",
    opticalROTR32(a, 7),
    (a >> 7) | (a << 25)
  );

  Serial.println();
  Serial.println(F("Self-test complete."));

  Serial.print(F("Errors="));
  Serial.println(errorCount);
}


// ============================================================
// Operation selection
// ============================================================

bool setOperationByName(const char* name)
{
  if (strcmp(name, "AND") == 0) {
    currentOperation = OP_AND;
  } else if (strcmp(name, "OR") == 0) {
    currentOperation = OP_OR;
  } else if (strcmp(name, "XOR") == 0) {
    currentOperation = OP_XOR;
  } else if (strcmp(name, "NAND") == 0) {
    currentOperation = OP_NAND;
  } else if (strcmp(name, "NOR") == 0) {
    currentOperation = OP_NOR;
  } else if (strcmp(name, "XNOR") == 0) {
    currentOperation = OP_XNOR;
  } else if (strcmp(name, "IMPLIES") == 0) {
    currentOperation = OP_IMPLIES;
  } else if (strcmp(name, "NIMPLY") == 0) {
    currentOperation = OP_NIMPLY;
  } else if (strcmp(name, "NOT") == 0) {
    currentOperation = OP_NOT;
  } else if (strcmp(name, "HALFADD") == 0) {
    currentOperation = OP_HALFADD;
  } else if (strcmp(name, "ADD32") == 0) {
    currentOperation = OP_ADD32;
  } else if (strcmp(name, "SHR") == 0) {
    currentOperation = OP_SHR;
  } else if (strcmp(name, "SHL") == 0) {
    currentOperation = OP_SHL;
  } else if (strcmp(name, "ROTR") == 0) {
    currentOperation = OP_ROTR;
  } else {
    return false;
  }

  strncpy(
    currentOperationName,
    name,
    sizeof(currentOperationName) - 1
  );

  currentOperationName[
    sizeof(currentOperationName) - 1
  ] = '\0';

  return true;
}


// ============================================================
// Status and help
// ============================================================

void printStatus()
{
  printSeparator();

  Serial.println(F("OPTICAL UNO STATUS"));

  Serial.print(F("Operation: "));
  Serial.println(currentOperationName);

  Serial.print(F("Operation amount: "));
  Serial.println(operationAmount);

  Serial.print(F("Threshold: "));
  Serial.println(photodiodeThreshold);

  Serial.print(F("Hysteresis: "));
  Serial.println(photodiodeHysteresis);

  Serial.print(F("Samples: "));
  Serial.println(sampleCount);

  Serial.print(F("Valve A level: "));
  Serial.println(valveALevel);

  Serial.print(F("Valve B level: "));
  Serial.println(valveBLevel);

  Serial.print(F("Valves enabled: "));
  Serial.println(
    valvesEnabled ? F("yes") : F("no")
  );

  Serial.print(F("A level: "));
  Serial.print(operandALevel);
  Serial.print(F(" bit="));
  printBit(operandABit);
  Serial.println();

  Serial.print(F("B level: "));
  Serial.print(operandBLevel);
  Serial.print(F(" bit="));
  printBit(operandBBit);
  Serial.println();

  Serial.print(F("Result level: "));
  Serial.print(resultLevel);
  Serial.print(F(" bit="));
  printBit(resultBit);
  Serial.println();

  Serial.print(F("Ambient: "));
  Serial.println(ambientLevel);

  Serial.print(F("Register A: "));
  printHex32(registerA);
  Serial.println();

  Serial.print(F("Register B: "));
  printHex32(registerB);
  Serial.println();

  Serial.print(F("Register result: "));
  printHex32(registerResult);
  Serial.println();

  Serial.print(F("Carry: "));
  printHex32(registerCarry);
  Serial.println();

  Serial.print(F("Optical events: "));
  Serial.println(opticalEvents);

  Serial.print(F("Operations: "));
  Serial.println(operationCount);

  Serial.print(F("Errors: "));
  Serial.println(errorCount);

  printSeparator();
}

void printHelp()
{
  printSeparator();

  Serial.println(F("COMMANDS"));

  Serial.println(F("help"));
  Serial.println(F("status"));
  Serial.println(F("read"));
  Serial.println(F("calibrate"));
  Serial.println(F("ambient"));
  Serial.println(F("threshold 500"));
  Serial.println(F("hysteresis 20"));
  Serial.println(F("samples 8"));

  Serial.println(F("operation AND"));
  Serial.println(F("operation OR"));
  Serial.println(F("operation XOR"));
  Serial.println(F("operation NAND"));
  Serial.println(F("operation NOR"));
  Serial.println(F("operation XNOR"));
  Serial.println(F("operation IMPLIES"));
  Serial.println(F("operation NIMPLY"));
  Serial.println(F("operation NOT"));
  Serial.println(F("operation HALFADD"));
  Serial.println(F("operation ADD32"));
  Serial.println(F("operation SHR 3"));
  Serial.println(F("operation SHL 3"));
  Serial.println(F("operation ROTR 7"));

  Serial.println(F("rega 0x12345678"));
  Serial.println(F("regb 0x87654321"));
  Serial.println(F("execute"));

  Serial.println(F("valvea 255"));
  Serial.println(F("valveb 255"));
  Serial.println(F("valves 255 128"));
  Serial.println(F("stop"));

  Serial.println(F("test"));
  Serial.println(F("sha256"));
  Serial.println(F("sha256 abc"));
  Serial.println(F("sha256 hello world"));

  Serial.println(F("trace on"));
  Serial.println(F("trace off"));
  Serial.println(F("stream on"));
  Serial.println(F("stream off"));

  printSeparator();
}


// ============================================================
// Command processing
// ============================================================

void processCommand(char* command)
{
  char* cursor = command;
  char* first = nextToken(cursor);

  if (first == nullptr) {
    return;
  }

  uppercaseInPlace(first);

  if (strcmp(first, "HELP") == 0) {
    printHelp();
    return;
  }

  if (strcmp(first, "STATUS") == 0) {
    sampleOpticalInputs();
    printStatus();
    return;
  }

  if (strcmp(first, "READ") == 0) {
    sampleOpticalInputs();
    printOpticalState();
    return;
  }

  if (strcmp(first, "CALIBRATE") == 0) {
    calibrateThreshold();
    return;
  }

  if (strcmp(first, "AMBIENT") == 0) {
    automaticAmbientCalibration();
    return;
  }

  if (strcmp(first, "THRESHOLD") == 0) {
    char* value = nextToken(cursor);

    if (value == nullptr) {
      Serial.print(F("Threshold="));
      Serial.println(photodiodeThreshold);
    } else {
      photodiodeThreshold =
        constrain(parseNumber(value), 0, 1023);

      Serial.print(F("Threshold="));
      Serial.println(photodiodeThreshold);
    }

    return;
  }

  if (strcmp(first, "HYSTERESIS") == 0) {
    char* value = nextToken(cursor);

    if (value == nullptr) {
      Serial.print(F("Hysteresis="));
      Serial.println(photodiodeHysteresis);
    } else {
      photodiodeHysteresis =
        constrain(parseNumber(value), 0, 500);

      Serial.print(F("Hysteresis="));
      Serial.println(photodiodeHysteresis);
    }

    return;
  }

  if (strcmp(first, "SAMPLES") == 0) {
    char* value = nextToken(cursor);

    if (value != nullptr) {
      sampleCount =
        constrain(parseNumber(value), 1, 32);
    }

    Serial.print(F("Samples="));
    Serial.println(sampleCount);

    return;
  }

  if (strcmp(first, "OPERATION") == 0) {
    char* operation = nextToken(cursor);

    if (operation == nullptr) {
      Serial.print(F("Operation="));
      Serial.println(currentOperationName);
      return;
    }

    uppercaseInPlace(operation);

    if (!setOperationByName(operation)) {
      Serial.println(F("Unknown operation."));
      return;
    }

    char* amount = nextToken(cursor);

    if (amount != nullptr) {
      operationAmount =
        constrain(parseNumber(amount), 0, 31);
    } else {
      operationAmount = 0;
    }

    Serial.print(F("Operation="));
    Serial.print(currentOperationName);

    Serial.print(F(" amount="));
    Serial.println(operationAmount);

    return;
  }

  if (strcmp(first, "REGA") == 0) {
    char* value = nextToken(cursor);

    if (value != nullptr) {
      registerA =
        parseNumber(value);
    }

    Serial.print(F("Register A="));
    printHex32(registerA);
    Serial.println();

    return;
  }

  if (strcmp(first, "REGB") == 0) {
    char* value = nextToken(cursor);

    if (value != nullptr) {
      registerB =
        parseNumber(value);
    }

    Serial.print(F("Register B="));
    printHex32(registerB);
    Serial.println();

    return;
  }

  if (strcmp(first, "EXECUTE") == 0) {
    executeOpticalOperation();
    return;
  }

  if (strcmp(first, "TEST") == 0) {
    runSelfTest();
    return;
  }

  if (strcmp(first, "SHA256") == 0) {
    char* message =
      skipSpaces(cursor);

    if (*message == '\0') {
      runSha256Test();
    } else {
      sha256Command(message);
    }

    return;
  }

  if (strcmp(first, "VALVEA") == 0) {
    char* value =
      nextToken(cursor);

    if (value != nullptr) {
      setValveA(
        constrain(parseNumber(value), 0, 255)
      );

      valvesEnable();
    }

    Serial.print(F("Valve A="));
    Serial.println(valveALevel);

    return;
  }

  if (strcmp(first, "VALVEB") == 0) {
    char* value =
      nextToken(cursor);

    if (value != nullptr) {
      setValveB(
        constrain(parseNumber(value), 0, 255)
      );

      valvesEnable();
    }

    Serial.print(F("Valve B="));
    Serial.println(valveBLevel);

    return;
  }

  if (strcmp(first, "VALVES") == 0) {
    char* valueA =
      nextToken(cursor);

    char* valueB =
      nextToken(cursor);

    if (valueA != nullptr) {
      valveALevel =
        constrain(parseNumber(valueA), 0, 255);
    }

    if (valueB != nullptr) {
      valveBLevel =
        constrain(parseNumber(valueB), 0, 255);
    }

    setBothValves(
      valveALevel,
      valveBLevel
    );

    Serial.print(F("Valve A="));
    Serial.print(valveALevel);

    Serial.print(F(" Valve B="));
    Serial.println(valveBLevel);

    return;
  }

  if (strcmp(first, "STOP") == 0) {
    valvesOff();
    Serial.println(F("Valves stopped."));
    return;
  }

  if (strcmp(first, "TRACE") == 0) {
    char* value =
      nextToken(cursor);

    if (value != nullptr) {
      uppercaseInPlace(value);

      traceEnabled =
        strcmp(value, "ON") == 0;
    }

    Serial.print(F("Trace="));
    Serial.println(
      traceEnabled ? F("on") : F("off")
    );

    return;
  }

  if (strcmp(first, "STREAM") == 0) {
    char* value =
      nextToken(cursor);

    if (value != nullptr) {
      uppercaseInPlace(value);

      streamEnabled =
        strcmp(value, "ON") == 0;
    }

    Serial.print(F("Stream="));
    Serial.println(
      streamEnabled ? F("on") : F("off")
    );

    return;
  }

  Serial.println(
    F("Unknown command. Type help.")
  );
}


// ============================================================
// Serial input
// ============================================================

void readSerialCommands()
{
  while (Serial.available()) {
    char incoming =
      Serial.read();

    if (incoming == '\r') {
      continue;
    }

    if (incoming == '\n') {
      commandBuffer[commandLength] =
        '\0';

      processCommand(commandBuffer);

      commandLength = 0;
      continue;
    }

    if (commandLength <
        COMMAND_BUFFER_SIZE - 1) {
      commandBuffer[commandLength++] =
        incoming;
    }
  }
}


// ============================================================
// Stream mode
// ============================================================

void streamOpticalState()
{
  static uint32_t lastStreamTime = 0;

  if (!streamEnabled) {
    return;
  }

  if (millis() - lastStreamTime < 250) {
    return;
  }

  lastStreamTime = millis();

  sampleOpticalInputs();

  Serial.print(F("STREAM,"));
  Serial.print(millis());

  Serial.print(',');
  Serial.print(operandALevel);

  Serial.print(',');
  Serial.print(operandBLevel);

  Serial.print(',');
  Serial.print(resultLevel);

  Serial.print(',');
  Serial.print(ambientLevel);

  Serial.print(',');
  printBit(operandABit);

  Serial.print(',');
  printBit(operandBBit);

  Serial.print(',');
  printBit(resultBit);

  Serial.print(',');
  Serial.print(valveALevel);

  Serial.print(',');
  Serial.println(valveBLevel);
}


// ============================================================
// Setup
// ============================================================

void setup()
{
  pinMode(MUX_S0, OUTPUT);
  pinMode(MUX_S1, OUTPUT);
  pinMode(MUX_S2, OUTPUT);
  pinMode(MUX_ENABLE, OUTPUT);

  pinMode(VALVE_A_PWM_PIN, OUTPUT);
  pinMode(VALVE_B_PWM_PIN, OUTPUT);
  pinMode(VALVES_ENABLE_PIN, OUTPUT);

  digitalWrite(MUX_ENABLE, LOW);

  valvesOff();

  Serial.begin(115200);

  delay(500);

  Serial.println();
  Serial.println(
    F("Arduino Uno Two-Valve Optical Processor")
  );

  Serial.println(
    F("D5=Valve A PWM, D6=Valve B PWM, D4=enable")
  );

  Serial.println(
    F("Type help for commands.")
  );

  automaticAmbientCalibration();
  sampleOpticalInputs();
}


// ============================================================
// Main loop
// ============================================================

void loop()
{
  readSerialCommands();
  streamOpticalState();
}
