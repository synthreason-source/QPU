#include <Arduino.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

// ============================================================
// Dynamic real-optical Arduino Uno processor
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
// Photodiode channels:
//
//   Y0 -> operand photodiode 0
//   Y1 -> operand photodiode 1
//   Y2 -> result photodiode
//   Y3 -> ambient photodiode
//
// Default valve PWM channels:
//
//   Valve[0] -> D3 PWM
//   Valve[1] -> D5 PWM
//
// Common valve-driver enable:
//
//   D4
//
// Serial speed:
//
//   115200
//
// Important:
//
//   Each photodiode should use a transimpedance amplifier.
//   Do not connect a bare photodiode directly when accurate
//   measurements are required.
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
//   valves
//   valvecount 2
//   valvepin 0 3
//   valvepin 1 5
//   valvelevel 0 255
//   valve 0 1
//   valve 1 0
//   stop
//
//   mode optical
//   mode software
//
//   operation AND
//   operation OR
//   operation XOR
//   operation NAND
//   operation NOR
//   operation XNOR
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
//   sha256
//   sha256 abc
//   sha256mode optical
//   sha256mode software
//
//   test
//   trace on
//   trace off
//   stream on
//   stream off
//
// ============================================================


// ============================================================
// Fixed hardware pins
// ============================================================

const uint8_t ADC_PIN = A0;

const uint8_t MUX_S0 = 7;
const uint8_t MUX_S1 = 8;
const uint8_t MUX_S2 = 9;
const uint8_t MUX_ENABLE = 10;

const uint8_t COMMON_VALVE_ENABLE_PIN = 4;


// ============================================================
// Runtime valve configuration
// ============================================================

const uint8_t MAX_VALVE_SYSTEMS = 6;

uint8_t activeValveCount = 2;

uint8_t valvePwmPins[MAX_VALVE_SYSTEMS] = {
  3,
  5,
  6,
  11,
  9,
  10
};

uint8_t valveLevels[MAX_VALVE_SYSTEMS] = {
  255,
  255,
  255,
  255,
  255,
  255
};

bool valveStates[MAX_VALVE_SYSTEMS] = {
  false,
  false,
  false,
  false,
  false,
  false
};

bool valvesEnabled = false;


// ============================================================
// Photodiode channels
// ============================================================

const uint8_t MUX_INPUT_0 = 0;
const uint8_t MUX_INPUT_1 = 1;
const uint8_t MUX_RESULT = 2;
const uint8_t MUX_AMBIENT = 3;

const uint8_t INPUT_COUNT = 2;


// ============================================================
// General constants
// ============================================================

const uint16_t ADC_MAX_VALUE = 1023;

const uint16_t DEFAULT_THRESHOLD = 500;
const uint16_t DEFAULT_HYSTERESIS = 20;

const uint8_t DEFAULT_SAMPLE_COUNT = 8;

const uint16_t MUX_SETTLE_TIME_US = 150;
const uint16_t VALVE_SETTLE_TIME_US = 300;

const uint16_t OPTICAL_PRECHARGE_TIME_US = 100;
const uint16_t OPTICAL_SAMPLE_DELAY_US = 100;

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
// Operation and execution modes
// ============================================================

enum OperationCode {
  OP_AND,
  OP_OR,
  OP_XOR,
  OP_NAND,
  OP_NOR,
  OP_XNOR,
  OP_NOT,
  OP_HALFADD,
  OP_ADD32,
  OP_SHR,
  OP_SHL,
  OP_ROTR
};

enum ExecutionMode {
  MODE_OPTICAL,
  MODE_SOFTWARE
};

ExecutionMode executionMode = MODE_OPTICAL;


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
  0x27B70A85UL, 0x2E1B2138UL,
  0x4D2C6DFCUL, 0x53380D13UL,
  0x650A7354UL, 0x766A0ABBUL,
  0x81C2C92EUL, 0x92722C85UL,
  0xA2BFE8A1UL, 0xA81A664BUL,
  0xC24B8B70UL, 0xC76C51A3UL,
  0xD192E819UL, 0xD6990624UL,
  0xF40E3585UL, 0x106AA070UL,
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

uint16_t inputLevels[INPUT_COUNT] = {
  0,
  0
};

bool inputBits[INPUT_COUNT] = {
  false,
  false
};

uint16_t resultLevel = 0;
uint16_t ambientLevel = 0;

bool resultBit = false;

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

bool isPwmPin(uint8_t pin);
bool validValveIndex(uint8_t index);
bool pinAlreadyUsed(uint8_t pin, uint8_t exceptIndex);
bool configureValveCount(uint8_t count);
bool configureValvePin(uint8_t index, uint8_t pin);
void initializeValveSystems();
void printValveConfiguration();
void enableValveSystems();
void disableValveSystems();
void setValveLevel(uint8_t index, uint8_t level);
void setValveState(uint8_t index, bool state);
void updateValveEnable();
void turnOffAllValves();
void driveIndexedBits(const bool* bits, uint8_t count);

void muxSelect(uint8_t channel);
uint16_t readMuxRaw(uint8_t channel);
uint16_t readPhotodiode(uint8_t channel);
bool photodiodeToBit(uint16_t level);
void sampleOpticalInputs();
void printOpticalState();

bool softwareAND(bool a, bool b);
bool softwareOR(bool a, bool b);
bool softwareXOR(bool a, bool b);
bool softwareNOT(bool a);

bool opticalSampleResult();
bool opticalBinaryOperation(
  bool a,
  bool b,
  OperationCode operation
);

bool executeSingleOperation(
  bool a,
  bool b
);

HalfAdderResult opticalHalfAdder(
  bool a,
  bool b
);

FullAdderResult opticalFullAdder(
  bool a,
  bool b,
  bool carryIn
);

uint32_t opticalAND32(uint32_t a, uint32_t b);
uint32_t opticalOR32(uint32_t a, uint32_t b);
uint32_t opticalXOR32(uint32_t a, uint32_t b);
uint32_t opticalNOT32(uint32_t value);
uint32_t opticalADD32(uint32_t a, uint32_t b);
uint32_t opticalSHR32(uint32_t value, uint8_t amount);
uint32_t opticalSHL32(uint32_t value, uint8_t amount);
uint32_t opticalROTR32(uint32_t value, uint8_t amount);

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

uint32_t sha256Ch(uint32_t x, uint32_t y, uint32_t z);
uint32_t sha256Maj(uint32_t x, uint32_t y, uint32_t z);
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
// Utility functions
// ============================================================

void incrementOpticalEvents()
{
  if (opticalEvents < MAX_OPTICAL_EVENTS) {
    opticalEvents++;
  }
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
// Runtime valve configuration
// ============================================================

bool isPwmPin(uint8_t pin)
{
  return pin == 3 ||
         pin == 5 ||
         pin == 6 ||
         pin == 9 ||
         pin == 10 ||
         pin == 11;
}

bool validValveIndex(uint8_t index)
{
  return index < activeValveCount &&
         index < MAX_VALVE_SYSTEMS;
}

bool pinAlreadyUsed(
  uint8_t pin,
  uint8_t exceptIndex
)
{
  for (uint8_t index = 0;
       index < activeValveCount;
       index++) {

    if (index == exceptIndex) {
      continue;
    }

    if (valvePwmPins[index] == pin) {
      return true;
    }
  }

  return false;
}

bool configureValveCount(uint8_t count)
{
  if (count == 0 ||
      count > MAX_VALVE_SYSTEMS) {
    return false;
  }

  for (uint8_t index = 0;
       index < count;
       index++) {

    if (!isPwmPin(valvePwmPins[index])) {
      return false;
    }

    if (pinAlreadyUsed(
          valvePwmPins[index],
          index
        )) {
      return false;
    }
  }

  turnOffAllValves();

  activeValveCount = count;

  initializeValveSystems();

  return true;
}

bool configureValvePin(
  uint8_t index,
  uint8_t pin
)
{
  if (index >= MAX_VALVE_SYSTEMS) {
    return false;
  }

  if (!isPwmPin(pin)) {
    return false;
  }

  if (pinAlreadyUsed(pin, index)) {
    return false;
  }

  analogWrite(
    valvePwmPins[index],
    0
  );

  valvePwmPins[index] = pin;

  pinMode(
    pin,
    OUTPUT
  );

  analogWrite(
    pin,
    0
  );

  valveStates[index] = false;

  updateValveEnable();

  return true;
}

void initializeValveSystems()
{
  pinMode(
    COMMON_VALVE_ENABLE_PIN,
    OUTPUT
  );

  digitalWrite(
    COMMON_VALVE_ENABLE_PIN,
    LOW
  );

  for (uint8_t index = 0;
       index < MAX_VALVE_SYSTEMS;
       index++) {

    pinMode(
      valvePwmPins[index],
      OUTPUT
    );

    analogWrite(
      valvePwmPins[index],
      0
    );

    valveStates[index] = false;
  }

  valvesEnabled = false;
}

void printValveConfiguration()
{
  Serial.print(F("Active valve count="));
  Serial.println(activeValveCount);

  for (uint8_t index = 0;
       index < activeValveCount;
       index++) {

    Serial.print(F("Valve["));
    Serial.print(index);

    Serial.print(F("] pin="));
    Serial.print(valvePwmPins[index]);

    Serial.print(F(" level="));
    Serial.print(valveLevels[index]);

    Serial.print(F(" state="));
    Serial.println(
      valveStates[index]
      ? F("on")
      : F("off")
    );
  }
}

void enableValveSystems()
{
  digitalWrite(
    COMMON_VALVE_ENABLE_PIN,
    HIGH
  );

  valvesEnabled = true;
}

void disableValveSystems()
{
  digitalWrite(
    COMMON_VALVE_ENABLE_PIN,
    LOW
  );

  valvesEnabled = false;
}

void setValveLevel(
  uint8_t index,
  uint8_t level
)
{
  if (!validValveIndex(index)) {
    return;
  }

  valveLevels[index] = level;

  analogWrite(
    valvePwmPins[index],
    level
  );

  valveStates[index] =
    level > 0;
}

void setValveState(
  uint8_t index,
  bool state
)
{
  if (!validValveIndex(index)) {
    return;
  }

  if (state) {
    analogWrite(
      valvePwmPins[index],
      valveLevels[index]
    );

    valveStates[index] =
      valveLevels[index] > 0;
  } else {
    analogWrite(
      valvePwmPins[index],
      0
    );

    valveStates[index] = false;
  }
}

void updateValveEnable()
{
  bool anyActive = false;

  for (uint8_t index = 0;
       index < activeValveCount;
       index++) {

    if (valveStates[index]) {
      anyActive = true;
      break;
    }
  }

  if (anyActive) {
    enableValveSystems();
  } else {
    disableValveSystems();
  }
}

void turnOffAllValves()
{
  for (uint8_t index = 0;
       index < activeValveCount;
       index++) {

    analogWrite(
      valvePwmPins[index],
      0
    );

    valveStates[index] = false;
  }

  disableValveSystems();
}

void driveIndexedBits(
  const bool* bits,
  uint8_t count
)
{
  uint8_t limit = count;

  if (limit > activeValveCount) {
    limit = activeValveCount;
  }

  for (uint8_t index = 0;
       index < limit;
       index++) {

    setValveState(
      index,
      bits[index]
    );
  }

  for (uint8_t index = limit;
       index < activeValveCount;
       index++) {

    setValveState(
      index,
      false
    );
  }

  updateValveEnable();
}


// ============================================================
// Multiplexer and photodiode input
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
  digitalWrite(
    MUX_ENABLE,
    LOW
  );

  muxSelect(channel);

  uint32_t total = 0;

  for (uint8_t i = 0;
       i < sampleCount;
       i++) {

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

void sampleOpticalInputs()
{
  inputLevels[0] =
    readPhotodiode(MUX_INPUT_0);

  inputLevels[1] =
    readPhotodiode(MUX_INPUT_1);

  inputBits[0] =
    photodiodeToBit(inputLevels[0]);

  inputBits[1] =
    photodiodeToBit(inputLevels[1]);

  resultLevel =
    readPhotodiode(MUX_RESULT);

  ambientLevel =
    readPhotodiode(MUX_AMBIENT);

  resultBit =
    photodiodeToBit(resultLevel);
}

void printOpticalState()
{
  for (uint8_t index = 0;
       index < INPUT_COUNT;
       index++) {

    Serial.print(F("INPUT["));
    Serial.print(index);

    Serial.print(F("]="));
    Serial.print(inputLevels[index]);

    Serial.print(F(":"));
    printBit(inputBits[index]);

    Serial.print(' ');
  }

  Serial.print(F("RESULT="));
  Serial.print(resultLevel);

  Serial.print(F(":"));
  printBit(resultBit);

  Serial.print(F(" AMBIENT="));
  Serial.println(ambientLevel);
}


// ============================================================
// Software primitives
// ============================================================

bool softwareAND(bool a, bool b)
{
  return a && b;
}

bool softwareOR(bool a, bool b)
{
  return a || b;
}

bool softwareXOR(bool a, bool b)
{
  return a ^ b;
}

bool softwareNOT(bool a)
{
  return !a;
}


// ============================================================
// Real optical operation
// ============================================================

bool opticalSampleResult()
{
  delayMicroseconds(
    OPTICAL_SAMPLE_DELAY_US
  );

  uint16_t measured =
    readPhotodiode(MUX_RESULT);

  resultLevel = measured;

  bool bit =
    photodiodeToBit(measured);

  resultBit = bit;

  return bit;
}

bool opticalBinaryOperation(
  bool a,
  bool b,
  OperationCode operation
)
{
  incrementOpticalEvents();

  // This assumes that the external optical hardware itself
  // implements the selected operation. The Arduino only places
  // the two operands into the two indexed optical channels.
  //
  // For example, for an AND optical gate:
  //
  //   Valve[0] -> optical input A
  //   Valve[1] -> optical input B
  //   physical optical gate -> MUX_RESULT photodiode
  //
  // The Arduino reads the resulting light level.

  bool bits[2] = {
    a,
    b
  };

  driveIndexedBits(
    bits,
    2
  );

  delayMicroseconds(
    VALVE_SETTLE_TIME_US
  );

  bool measured =
    opticalSampleResult();

  turnOffAllValves();

  if (traceEnabled) {
    Serial.print(F("OPTICAL("));
    printBit(a);
    Serial.print(',');
    printBit(b);
    Serial.print(F(")->"));
    printBit(measured);
    Serial.print(F(" ADC="));
    Serial.println(resultLevel);
  }

  return measured;
}

bool executeSingleOperation(
  bool a,
  bool b
)
{
  if (executionMode == MODE_SOFTWARE) {
    switch (currentOperation) {
      case OP_AND:
        return softwareAND(a, b);

      case OP_OR:
        return softwareOR(a, b);

      case OP_XOR:
        return softwareXOR(a, b);

      case OP_NAND:
        return !softwareAND(a, b);

      case OP_NOR:
        return !softwareOR(a, b);

      case OP_XNOR:
        return !softwareXOR(a, b);

      case OP_NOT:
        return softwareNOT(a);

      default:
        return false;
    }
  }

  switch (currentOperation) {
    case OP_AND:
    case OP_OR:
    case OP_XOR:
    case OP_NAND:
    case OP_NOR:
    case OP_XNOR:
      return opticalBinaryOperation(
        a,
        b,
        currentOperation
      );

    case OP_NOT: {
      bool bits[2] = {
        a,
        false
      };

      driveIndexedBits(
        bits,
        1
      );

      delayMicroseconds(
        VALVE_SETTLE_TIME_US
      );

      bool measured =
        opticalSampleResult();

      turnOffAllValves();

      return measured;
    }

    default:
      return false;
  }
}


// ============================================================
// Optical adders
// ============================================================
//
// These functions are real optical only if the external optical
// hardware supports the requested gate mode and the result
// photodiode is physically connected to that gate's output.
//
// For a real full adder, the optical apparatus must be switched
// between the required two-input suboperations.
//

HalfAdderResult opticalHalfAdder(
  bool a,
  bool b
)
{
  HalfAdderResult result;

  if (executionMode == MODE_SOFTWARE) {
    result.sum =
      softwareXOR(a, b);

    result.carry =
      softwareAND(a, b);

    return result;
  }

  // Physical hardware must be configured for XOR.
  currentOperation = OP_XOR;
  result.sum =
    opticalBinaryOperation(
      a,
      b,
      OP_XOR
    );

  // Physical hardware must then be configured for AND.
  currentOperation = OP_AND;
  result.carry =
    opticalBinaryOperation(
      a,
      b,
      OP_AND
    );

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
    opticalHalfAdder(
      a,
      b
    );

  second =
    opticalHalfAdder(
      first.sum,
      carryIn
    );

  if (executionMode == MODE_SOFTWARE) {
    result.sum =
      second.sum;

    result.carry =
      first.carry ||
      second.carry;

    return result;
  }

  currentOperation = OP_OR;

  result.sum =
    second.sum;

  result.carry =
    opticalBinaryOperation(
      first.carry,
      second.carry,
      OP_OR
    );

  return result;
}


// ============================================================
// 32-bit optical operations
// ============================================================

uint32_t opticalAND32(
  uint32_t a,
  uint32_t b
)
{
  uint32_t result = 0;

  for (uint8_t bit = 0;
       bit < 32;
       bit++) {

    bool aBit =
      (a >> bit) & 1UL;

    bool bBit =
      (b >> bit) & 1UL;

    bool output =
      executionMode == MODE_SOFTWARE
      ? softwareAND(aBit, bBit)
      : opticalBinaryOperation(
          aBit,
          bBit,
          OP_AND
        );

    if (output) {
      result |= (1UL << bit);
    }
  }

  return result;
}

uint32_t opticalOR32(
  uint32_t a,
  uint32_t b
)
{
  uint32_t result = 0;

  for (uint8_t bit = 0;
       bit < 32;
       bit++) {

    bool aBit =
      (a >> bit) & 1UL;

    bool bBit =
      (b >> bit) & 1UL;

    bool output =
      executionMode == MODE_SOFTWARE
      ? softwareOR(aBit, bBit)
      : opticalBinaryOperation(
          aBit,
          bBit,
          OP_OR
        );

    if (output) {
      result |= (1UL << bit);
    }
  }

  return result;
}

uint32_t opticalXOR32(
  uint32_t a,
  uint32_t b
)
{
  uint32_t result = 0;

  for (uint8_t bit = 0;
       bit < 32;
       bit++) {

    bool aBit =
      (a >> bit) & 1UL;

    bool bBit =
      (b >> bit) & 1UL;

    bool output =
      executionMode == MODE_SOFTWARE
      ? softwareXOR(aBit, bBit)
      : opticalBinaryOperation(
          aBit,
          bBit,
          OP_XOR
        );

    if (output) {
      result |= (1UL << bit);
    }
  }

  return result;
}

uint32_t opticalNOT32(uint32_t value)
{
  uint32_t result = 0;

  for (uint8_t bit = 0;
       bit < 32;
       bit++) {

    bool input =
      (value >> bit) & 1UL;

    bool output =
      executionMode == MODE_SOFTWARE
      ? softwareNOT(input)
      : executeSingleOperation(
          input,
          false
        );

    if (output) {
      result |= (1UL << bit);
    }
  }

  return result;
}

uint32_t opticalADD32(
  uint32_t a,
  uint32_t b
)
{
  if (executionMode == MODE_SOFTWARE) {
    registerCarry =
      (a + b) < a
      ? 1UL
      : 0UL;

    return a + b;
  }

  uint32_t result = 0;
  bool carry = false;

  for (uint8_t bit = 0;
       bit < 32;
       bit++) {

    bool aBit =
      (a >> bit) & 1UL;

    bool bBit =
      (b >> bit) & 1UL;

    FullAdderResult fullAdder =
      opticalFullAdder(
        aBit,
        bBit,
        carry
      );

    if (fullAdder.sum) {
      result |= (1UL << bit);
    }

    carry =
      fullAdder.carry;
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

void executeOpticalOperation()
{
  sampleOpticalInputs();
  operationCount++;

  bool a =
    inputBits[0];

  bool b =
    inputBits[1];

  if (currentOperation == OP_HALFADD) {
    HalfAdderResult result =
      opticalHalfAdder(
        a,
        b
      );

    Serial.print(F("HALFADD SUM="));
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
      a,
      b
    );

  Serial.print(currentOperationName);

  Serial.print(F(" A="));
  printBit(a);

  Serial.print(F(" B="));
  printBit(b);

  Serial.print(F(" OUT="));
  printBit(output);

  Serial.print(F(" RESULT_ADC="));
  Serial.println(resultLevel);
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

  return opticalXOR32(
    xy,
    notXz
  );
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

  return opticalXOR32(
    opticalXOR32(xy, xz),
    yz
  );
}

uint32_t sha256BigSigma0(uint32_t x)
{
  return opticalXOR32(
    opticalXOR32(
      opticalROTR32(x, 2),
      opticalROTR32(x, 13)
    ),
    opticalROTR32(x, 22)
  );
}

uint32_t sha256BigSigma1(uint32_t x)
{
  return opticalXOR32(
    opticalXOR32(
      opticalROTR32(x, 6),
      opticalROTR32(x, 11)
    ),
    opticalROTR32(x, 25)
  );
}

uint32_t sha256SmallSigma0(uint32_t x)
{
  return opticalXOR32(
    opticalXOR32(
      opticalROTR32(x, 7),
      opticalROTR32(x, 18)
    ),
    opticalSHR32(x, 3)
  );
}

uint32_t sha256SmallSigma1(uint32_t x)
{
  return opticalXOR32(
    opticalXOR32(
      opticalROTR32(x, 17),
      opticalROTR32(x, 19)
    ),
    opticalSHR32(x, 10)
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
    state[i] =
      SHA256_H0[i];
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

    for (uint8_t i = 0;
         i < SHA256_BLOCK_SIZE;
         i++) {

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
      uint32_t s0 =
        sha256SmallSigma0(w[t - 15]);

      uint32_t s1 =
        sha256SmallSigma1(w[t - 2]);

      uint32_t p1 =
        opticalADD32(
          w[t - 16],
          s0
        );

      uint32_t p2 =
        opticalADD32(
          p1,
          w[t - 7]
        );

      w[t] =
        opticalADD32(
          p2,
          s1
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
      uint32_t s1 =
        sha256BigSigma1(e);

      uint32_t ch =
        sha256Ch(e, f, g);

      uint32_t temp1 =
        opticalADD32(h, s1);

      temp1 =
        opticalADD32(temp1, ch);

      temp1 =
        opticalADD32(
          temp1,
          SHA256_K[t]
        );

      temp1 =
        opticalADD32(
          temp1,
          w[t]
        );

      uint32_t s0 =
        sha256BigSigma0(a);

      uint32_t maj =
        sha256Maj(a, b, c);

      uint32_t temp2 =
        opticalADD32(s0, maj);

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
  for (uint8_t i = 0;
       i < SHA256_DIGEST_SIZE;
       i++) {

    if (digest[i] < 0x10) {
      Serial.print('0');
    }

    Serial.print(
      digest[i],
      HEX
    );
  }

  Serial.println();
}

void sha256Command(
  const char* message
)
{
  uint16_t length =
    strlen(message);

  if (length > 55) {
    Serial.println(
      F("Use 55 bytes or less on Uno.")
    );

    return;
  }

  uint8_t digest[SHA256_DIGEST_SIZE];

  opticalEvents = 0;

  Serial.print(F("SHA256 mode="));

  Serial.println(
    executionMode == MODE_OPTICAL
    ? F("OPTICAL")
    : F("SOFTWARE")
  );

  Serial.print(F("Input="));
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

  Serial.print(F("SHA256="));
  printSha256Digest(digest);

  Serial.print(F("Time_ms="));
  Serial.println(elapsed);

  Serial.print(F("Optical events="));
  Serial.println(opticalEvents);
}

void runSha256Test()
{
  static const char message[] =
    "abc";

  static const uint8_t expected[
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

  uint8_t actual[
    SHA256_DIGEST_SIZE
  ];

  Serial.println(
    F("SHA-256 test: abc")
  );

  uint32_t startTime =
    millis();

  sha256Hash(
    (const uint8_t*)message,
    3,
    actual
  );

  uint32_t elapsed =
    millis() - startTime;

  Serial.print(F("Calculated="));
  printSha256Digest(actual);

  Serial.print(F("Expected="));
  printSha256Digest(expected);

  bool passed = true;

  for (uint8_t i = 0;
       i < SHA256_DIGEST_SIZE;
       i++) {

    if (actual[i] != expected[i]) {
      passed = false;
      break;
    }
  }

  Serial.print(F("Result="));
  Serial.println(
    passed
    ? F("PASS")
    : F("FAIL")
  );

  Serial.print(F("Time_ms="));
  Serial.println(elapsed);

  Serial.print(F("Optical events="));
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
  turnOffAllValves();

  Serial.println(
    F("Block the result photodiode.")
  );

  Serial.println(
    F("Press a key for dark measurement.")
  );

  while (!Serial.available()) {
    delay(10);
  }

  while (Serial.available()) {
    Serial.read();
  }

  uint16_t dark =
    readPhotodiode(MUX_RESULT);

  Serial.println(
    F("Illuminate the result photodiode.")
  );

  Serial.println(
    F("Press a key for bright measurement.")
  );

  while (!Serial.available()) {
    delay(10);
  }

  while (Serial.available()) {
    Serial.read();
  }

  uint16_t bright =
    readPhotodiode(MUX_RESULT);

  if (bright >= dark) {
    photodiodeThreshold =
      dark + ((bright - dark) / 2);
  } else {
    photodiodeThreshold =
      bright + ((dark - bright) / 2);
  }

  Serial.print(F("Dark="));
  Serial.println(dark);

  Serial.print(F("Bright="));
  Serial.println(bright);

  Serial.print(F("Threshold="));
  Serial.println(photodiodeThreshold);
}

void automaticAmbientCalibration()
{
  turnOffAllValves();

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
// Self-tests
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

  Serial.println(
    F("Running software primitive tests.")
  );

  ExecutionMode savedMode =
    executionMode;

  executionMode =
    MODE_SOFTWARE;

  bool values[4][2] = {
    {false, false},
    {false, true},
    {true, false},
    {true, true}
  };

  for (uint8_t i = 0; i < 4; i++) {
    bool a = values[i][0];
    bool b = values[i][1];

    testBitOperation(
      "AND",
      softwareAND(a, b),
      a && b
    );

    testBitOperation(
      "OR",
      softwareOR(a, b),
      a || b
    );

    testBitOperation(
      "XOR",
      softwareXOR(a, b),
      a ^ b
    );

    testBitOperation(
      "NOT",
      softwareNOT(a),
      !a
    );
  }

  uint32_t a =
    0x12345678UL;

  uint32_t b =
    0x87654321UL;

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

  executionMode =
    savedMode;

  Serial.print(F("Errors="));
  Serial.println(errorCount);
}


// ============================================================
// Operation selection
// ============================================================

bool setOperationByName(
  const char* name
)
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

  Serial.println(
    F("REAL OPTICAL PROCESSOR STATUS")
  );

  Serial.print(F("Mode="));
  Serial.println(
    executionMode == MODE_OPTICAL
    ? F("OPTICAL")
    : F("SOFTWARE")
  );

  Serial.print(F("Operation="));
  Serial.println(currentOperationName);

  Serial.print(F("Threshold="));
  Serial.println(photodiodeThreshold);

  Serial.print(F("Hysteresis="));
  Serial.println(photodiodeHysteresis);

  Serial.print(F("Samples="));
  Serial.println(sampleCount);

  Serial.print(F("Valves enabled="));
  Serial.println(
    valvesEnabled
    ? F("yes")
    : F("no")
  );

  printValveConfiguration();

  for (uint8_t index = 0;
       index < INPUT_COUNT;
       index++) {

    Serial.print(F("Input["));
    Serial.print(index);

    Serial.print(F("] ADC="));
    Serial.print(inputLevels[index]);

    Serial.print(F(" bit="));
    printBit(inputBits[index]);

    Serial.println();
  }

  Serial.print(F("Result ADC="));
  Serial.print(resultLevel);

  Serial.print(F(" bit="));
  printBit(resultBit);

  Serial.println();

  Serial.print(F("Ambient="));
  Serial.println(ambientLevel);

  Serial.print(F("Optical events="));
  Serial.println(opticalEvents);

  Serial.print(F("Operations="));
  Serial.println(operationCount);

  Serial.print(F("Errors="));
  Serial.println(errorCount);

  printSeparator();
}

void printHelp()
{
  printSeparator();

  Serial.println(F("MODE"));
  Serial.println(F("mode optical"));
  Serial.println(F("mode software"));

  Serial.println(F("VALVES"));
  Serial.println(F("valves"));
  Serial.println(F("valvecount 2"));
  Serial.println(F("valvepin 0 3"));
  Serial.println(F("valvepin 1 5"));
  Serial.println(F("valvelevel 0 255"));
  Serial.println(F("valve 0 1"));
  Serial.println(F("valve 1 0"));
  Serial.println(F("stop"));

  Serial.println(F("SENSORS"));
  Serial.println(F("status"));
  Serial.println(F("read"));
  Serial.println(F("calibrate"));
  Serial.println(F("ambient"));
  Serial.println(F("threshold 500"));
  Serial.println(F("hysteresis 20"));
  Serial.println(F("samples 8"));

  Serial.println(F("OPERATIONS"));
  Serial.println(F("operation AND"));
  Serial.println(F("operation OR"));
  Serial.println(F("operation XOR"));
  Serial.println(F("operation NAND"));
  Serial.println(F("operation NOR"));
  Serial.println(F("operation XNOR"));
  Serial.println(F("operation NOT"));
  Serial.println(F("operation HALFADD"));
  Serial.println(F("operation ADD32"));
  Serial.println(F("operation SHR 3"));
  Serial.println(F("operation SHL 3"));
  Serial.println(F("operation ROTR 7"));

  Serial.println(F("REGISTERS"));
  Serial.println(F("rega 0x12345678"));
  Serial.println(F("regb 0x87654321"));
  Serial.println(F("execute"));

  Serial.println(F("SHA256"));
  Serial.println(F("sha256"));
  Serial.println(F("sha256 abc"));
  Serial.println(F("sha256mode optical"));
  Serial.println(F("sha256mode software"));

  Serial.println(F("DEBUG"));
  Serial.println(F("test"));
  Serial.println(F("trace on"));
  Serial.println(F("trace off"));
  Serial.println(F("stream on"));
  Serial.println(F("stream off"));

  printSeparator();
}


// ============================================================
// Command processing
// ============================================================

void processCommand(
  char* command
)
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

  if (strcmp(first, "MODE") == 0) {
    char* mode =
      nextToken(cursor);

    if (mode == nullptr) {
      Serial.println(
        executionMode == MODE_OPTICAL
        ? F("Mode=OPTICAL")
        : F("Mode=SOFTWARE")
      );

      return;
    }

    uppercaseInPlace(mode);

    if (strcmp(mode, "OPTICAL") == 0) {
      executionMode =
        MODE_OPTICAL;

      Serial.println(
        F("Mode=OPTICAL")
      );

      return;
    }

    if (strcmp(mode, "SOFTWARE") == 0) {
      executionMode =
        MODE_SOFTWARE;

      Serial.println(
        F("Mode=SOFTWARE")
      );

      return;
    }

    Serial.println(
      F("Use mode optical or mode software.")
    );

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
    char* value =
      nextToken(cursor);

    if (value != nullptr) {
      photodiodeThreshold =
        constrain(
          parseNumber(value),
          0,
          1023
        );
    }

    Serial.print(F("Threshold="));
    Serial.println(photodiodeThreshold);

    return;
  }

  if (strcmp(first, "HYSTERESIS") == 0) {
    char* value =
      nextToken(cursor);

    if (value != nullptr) {
      photodiodeHysteresis =
        constrain(
          parseNumber(value),
          0,
          500
        );
    }

    Serial.print(F("Hysteresis="));
    Serial.println(photodiodeHysteresis);

    return;
  }

  if (strcmp(first, "SAMPLES") == 0) {
    char* value =
      nextToken(cursor);

    if (value != nullptr) {
      sampleCount =
        constrain(
          parseNumber(value),
          1,
          32
        );
    }

    Serial.print(F("Samples="));
    Serial.println(sampleCount);

    return;
  }

  if (strcmp(first, "VALVES") == 0) {
    printValveConfiguration();
    return;
  }

  if (strcmp(first, "VALVECOUNT") == 0) {
    char* countText =
      nextToken(cursor);

    if (countText == nullptr) {
      Serial.print(F("Valve count="));
      Serial.println(activeValveCount);
      return;
    }

    uint8_t count =
      parseNumber(countText);

    if (!configureValveCount(count)) {
      Serial.println(
        F("Invalid valve count.")
      );
    } else {
      Serial.print(F("Valve count="));
      Serial.println(activeValveCount);
    }

    return;
  }

  if (strcmp(first, "VALVEPIN") == 0) {
    char* indexText =
      nextToken(cursor);

    char* pinText =
      nextToken(cursor);

    if (indexText == nullptr ||
        pinText == nullptr) {

      Serial.println(
        F("Usage: valvepin <index> <pin>")
      );

      return;
    }

    uint8_t index =
      parseNumber(indexText);

    uint8_t pin =
      parseNumber(pinText);

    if (!configureValvePin(index, pin)) {
      Serial.println(
        F("Invalid index, duplicate pin, or non-PWM pin.")
      );

      return;
    }

    Serial.print(F("Valve["));
    Serial.print(index);
    Serial.print(F("] pin="));
    Serial.println(pin);

    return;
  }

  if (strcmp(first, "VALVELEVEL") == 0) {
    char* indexText =
      nextToken(cursor);

    char* levelText =
      nextToken(cursor);

    if (indexText == nullptr ||
        levelText == nullptr) {

      Serial.println(
        F("Usage: valvelevel <index> <level>")
      );

      return;
    }

    uint8_t index =
      parseNumber(indexText);

    uint8_t level =
      constrain(
        parseNumber(levelText),
        0,
        255
      );

    if (!validValveIndex(index)) {
      Serial.println(
        F("Invalid valve index.")
      );

      return;
    }

    setValveLevel(
      index,
      level
    );

    updateValveEnable();

    Serial.print(F("Valve["));
    Serial.print(index);
    Serial.print(F("] level="));
    Serial.println(level);

    return;
  }

  if (strcmp(first, "VALVE") == 0) {
    char* indexText =
      nextToken(cursor);

    char* valueText =
      nextToken(cursor);

    if (indexText == nullptr ||
        valueText == nullptr) {

      Serial.println(
        F("Usage: valve <index> <0|1|level>")
      );

      return;
    }

    uint8_t index =
      parseNumber(indexText);

    uint16_t value =
      parseNumber(valueText);

    if (!validValveIndex(index)) {
      Serial.println(
        F("Invalid valve index.")
      );

      return;
    }

    if (value <= 1) {
      setValveState(
        index,
        value != 0
      );
    } else {
      setValveLevel(
        index,
        constrain(value, 0, 255)
      );
    }

    updateValveEnable();

    Serial.print(F("Valve["));
    Serial.print(index);

    Serial.print(F("] level="));
    Serial.print(valveLevels[index]);

    Serial.print(F(" state="));
    Serial.println(
      valveStates[index]
      ? F("on")
      : F("off")
    );

    return;
  }

  if (strcmp(first, "STOP") == 0) {
    turnOffAllValves();

    Serial.println(
      F("All valves stopped.")
    );

    return;
  }

  if (strcmp(first, "OPERATION") == 0) {
    char* operation =
      nextToken(cursor);

    if (operation == nullptr) {
      Serial.print(F("Operation="));
      Serial.println(currentOperationName);
      return;
    }

    uppercaseInPlace(operation);

    if (!setOperationByName(operation)) {
      Serial.println(
        F("Unknown operation.")
      );

      return;
    }

    char* amount =
      nextToken(cursor);

    if (amount != nullptr) {
      operationAmount =
        constrain(
          parseNumber(amount),
          0,
          31
        );
    } else {
      operationAmount = 0;
    }

    Serial.print(F("Operation="));
    Serial.println(currentOperationName);

    return;
  }

  if (strcmp(first, "REGA") == 0) {
    char* value =
      nextToken(cursor);

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
    char* value =
      nextToken(cursor);

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

  if (strcmp(first, "SHA256MODE") == 0) {
    char* mode =
      nextToken(cursor);

    if (mode == nullptr) {
      Serial.println(
        executionMode == MODE_OPTICAL
        ? F("SHA256 mode=OPTICAL")
        : F("SHA256 mode=SOFTWARE")
      );

      return;
    }

    uppercaseInPlace(mode);

    if (strcmp(mode, "OPTICAL") == 0) {
      executionMode =
        MODE_OPTICAL;

      Serial.println(
        F("SHA256 mode=OPTICAL")
      );

      return;
    }

    if (strcmp(mode, "SOFTWARE") == 0) {
      executionMode =
        MODE_SOFTWARE;

      Serial.println(
        F("SHA256 mode=SOFTWARE")
      );

      return;
    }

    Serial.println(
      F("Use sha256mode optical or sha256mode software.")
    );

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
      traceEnabled
      ? F("on")
      : F("off")
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
      streamEnabled
      ? F("on")
      : F("off")
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

  Serial.print(F(",MODE="));

  Serial.print(
    executionMode == MODE_OPTICAL
    ? F("OPTICAL")
    : F("SOFTWARE")
  );

  for (uint8_t index = 0;
       index < INPUT_COUNT;
       index++) {

    Serial.print(F(",IN"));
    Serial.print(index);
    Serial.print('=');
    Serial.print(inputLevels[index]);
    Serial.print(':');
    printBit(inputBits[index]);
  }

  Serial.print(F(",RESULT="));
  Serial.print(resultLevel);
  Serial.print(':');
  printBit(resultBit);

  Serial.print(F(",VALVES="));
  Serial.println(activeValveCount);
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

  digitalWrite(
    MUX_ENABLE,
    LOW
  );

  initializeValveSystems();

  Serial.begin(115200);

  delay(500);

  Serial.println();
  Serial.println(
    F("Arduino Uno Real Optical Processor")
  );

  Serial.println(
    F("Optical feedback mode available")
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
