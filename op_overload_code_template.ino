#include <Arduino.h>
#include <vector>
#include <string>

// Maximum limit for trace logs and loop safety
const int MAX_TRACE_STEPS = 256;
const int MAX_LOOP_ITERATIONS = 1000;

// Structure to hold trace log entries
struct TraceEntry {
    String op;
    int input1;
    int input2;
    int result;
    bool has_second_input;
};

// Global trace log container
std::vector<TraceEntry> global_trace;

void record_trace(String op, int in1, int in2, int res, bool two_inputs = true) {
    if (global_trace.size() < MAX_TRACE_STEPS) {
        global_trace.push_back({op, in1, in2, res, two_inputs});
    }
}

// =====================================================================
// OPTICAL PRIMITIVE SIMULATIONS (C++ Equivalents)
// =====================================================================

int opt_and(int x, int y) {
    int res = (x != 0 && y != 0) ? 1 : 0;
    record_trace("AND", x, y, res);
    return res;
}

int opt_or(int x, int y) {
    int res = (x != 0 || y != 0) ? 1 : 0;
    record_trace("OR", x, y, res);
    return res;
}

int opt_xor(int x, int y) {
    int res = (x != y) ? 1 : 0;
    record_trace("XOR", x, y, res);
    return res;
}

int opt_not(int x) {
    int res = (x == 0) ? 1 : 0;
    record_trace("NOT", x, 0, res, false);
    return res;
}

int optical_add(int a, int b) {
    int res = a + b;
    record_trace("ADD", a, b, res);
    return res;
}

int optical_subtract(int a, int b) {
    if (b > a) {
        // Subtractor error state
        return -1;
    }
    int res = a - b;
    record_trace("SUB", a, b, res);
    return res;
}

int optical_multiply(int a, int b) {
    int res = a * b;
    record_trace("MUL", a, b, res);
    return res;
}

int opt_cmp(int a, int b) {
    int forward = optical_subtract(a, b);
    if (forward >= 0) {
        return (forward == 0) ? 0 : 1;
    }
    int backward = optical_subtract(b, a);
    return (backward > 0) ? -1 : 0;
}

int opt_eq(int a, int b) { return (opt_cmp(a, b) == 0) ? 1 : 0; }
int opt_ne(int a, int b) { return (opt_cmp(a, b) != 0) ? 1 : 0; }
int opt_lt(int a, int b) { return (opt_cmp(a, b) < 0) ? 1 : 0; }
int opt_le(int a, int b) { return (opt_cmp(a, b) <= 0) ? 1 : 0; }
int opt_gt(int a, int b) { return (opt_cmp(a, b) > 0) ? 1 : 0; }
int opt_ge(int a, int b) { return (opt_cmp(a, b) >= 0) ? 1 : 0; }

// =====================================================================
// BIT CLASS (Operator Overloading for Embedded C++)
// =====================================================================

class Bit {
public:
    int value;

    Bit(int val) {
        value = (val < 0) ? 0 : val;
    }

    // Boolean operators
    Bit operator&(const Bit& other) const { return Bit(opt_and(value, other.value)); }
    Bit operator|(const Bit& other) const { return Bit(opt_or(value, other.value)); }
    Bit operator^(const Bit& other) const { return Bit(opt_xor(value, other.value)); }
    Bit operator~() const { return Bit(opt_not(value)); }

    // Arithmetic operators
    Bit operator+(const Bit& other) const { return Bit(optical_add(value, other.value)); }
    Bit operator-(const Bit& other) const { return Bit(optical_subtract(value, other.value)); }
    Bit operator*(const Bit& other) const { return Bit(optical_multiply(value, other.value)); }

    // Comparison operators
    bool operator==(const Bit& other) const { return opt_eq(value, other.value) == 1; }
    bool operator!=(const Bit& other) const { return opt_ne(value, other.value) == 1; }
    bool operator<(const Bit& other) const { return opt_lt(value, other.value) == 1; }
    bool operator<=(const Bit& other) const { return opt_le(value, other.value) == 1; }
    bool operator>(const Bit& other) const { return opt_gt(value, other.value) == 1; }
    bool operator>=(const Bit& other) const { return opt_ge(value, other.value) == 1; }
};

// =====================================================================
// ARDUINO SETUP & DEMO EXECUTION
// =====================================================================

void setup() {
    Serial.begin(115200);
    while (!Serial) {
        ; // Wait for serial port to connect (native USB boards)
    }

    delay(1000);
    Serial.println(F("\n--- Optical Evaluation Engine (Arduino Port) ---"));

    // Clear trace log before execution
    global_trace.clear();

    // Emulating the demo program:
    // target = 10
    // found = 0
    // for n in range(1, 20):
    //     if n * n == target:
    //         found = n
    //         break
    // print(found)

    int target = 10;
    int found = 0;
    int limit = 20;

    Serial.print(F("Searching for n where n * n == "));
    Serial.println(target);

    for (int n = 1; n < limit; n++) {
        Bit b_n(n);
        Bit b_target(target);
        
        // Emulating: n * n == target
        Bit product = b_n * b_n;
        if (product == b_target) {
            found = n;
            break;
        }
    }

    // Output results via Serial Monitor
    Serial.print(F("Execution Finished. Found value: "));
    Serial.println(found);

    Serial.println(F("\n--- Optical Operation Trace ---"));
    for (size_t i = 0; i < global_trace.size(); i++) {
        Serial.print(i + 1);
        Serial.print(F(": "));
        Serial.print(global_trace[i].op);
        Serial.print(F("("));
        Serial.print(global_trace[i].input1);
        if (global_trace[i].has_second_input) {
            Serial.print(F(", "));
            Serial.print(global_trace[i].input2);
        }
        Serial.print(F(") -> "));
        Serial.println(global_trace[i].result);
    }
}

void loop() {
    // Nothing to do in loop
}
