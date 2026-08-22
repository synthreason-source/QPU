/*
  ITO 32x32 Optical Bench Controller

  Receives from Python:

      BEGIN 32 32
      010101...
      101010...
      ...
      END

  The received binary pattern is held on the Arduino.

  IMPORTANT:
  This sketch contains the SERIAL PROTOCOL and a hardware
  abstraction for the ITO outputs.

  You MUST adapt setITOCell() to the actual electronics
  driving your ITO glass.

  The ITO glass itself should NOT be driven directly from
  Arduino GPIO pins.
*/

#include <Arduino.h>

#define ROWS 32
#define COLS 32

// ------------------------------------------------------------
// Current optical pattern
// ------------------------------------------------------------

uint8_t pattern[ROWS][COLS];

bool receivingPattern = false;

int currentRow = 0;

int expectedRows = 0;
int expectedCols = 0;


// ------------------------------------------------------------
// INITIALISE
// ------------------------------------------------------------

void setup()
{
    Serial.begin(115200);

    memset(
        pattern,
        0,
        sizeof(pattern)
    );

    /*
      Configure your actual ITO driver hardware here.

      Examples:

        shift registers
        SPI DACs
        analogue multiplexers
        row/column drivers
        external GPIO expanders
        DAC channels

      Do NOT connect a 50 mm ITO electrode directly
      to an Arduino pin if the electrical requirements
      exceed the Arduino's GPIO specifications.
    */

    initialiseITOHardware();

    Serial.println("ITO_READY");
}


// ------------------------------------------------------------
// MAIN LOOP
// ------------------------------------------------------------

void loop()
{
    if (Serial.available())
    {
        String line = Serial.readStringUntil('\n');

        line.trim();

        if (line.length() == 0)
            return;

        processLine(line);
    }
}


// ------------------------------------------------------------
// SERIAL PROTOCOL
// ------------------------------------------------------------

void processLine(String &line)
{
    // --------------------------------------------------------
    // BEGIN
    // --------------------------------------------------------

    if (line.startsWith("BEGIN"))
    {
        int firstSpace =
            line.indexOf(' ');

        int secondSpace =
            line.indexOf(
                ' ',
                firstSpace + 1
            );

        if (
            firstSpace < 0 ||
            secondSpace < 0
        )
        {
            Serial.println("ERROR BEGIN");
            return;
        }

        expectedRows =
            line.substring(
                firstSpace + 1,
                secondSpace
            ).toInt();

        expectedCols =
            line.substring(
                secondSpace + 1
            ).toInt();

        if (
            expectedRows != ROWS ||
            expectedCols != COLS
        )
        {
            Serial.println("ERROR DIMENSIONS");
            receivingPattern = false;
            return;
        }

        currentRow = 0;

        receivingPattern = true;

        Serial.println("BEGIN_OK");

        return;
    }


    // --------------------------------------------------------
    // END
    // --------------------------------------------------------

    if (line == "END")
    {
        if (!receivingPattern)
        {
            Serial.println("ERROR NO_PATTERN");
            return;
        }

        if (currentRow != ROWS)
        {
            Serial.println("ERROR ROW_COUNT");
            receivingPattern = false;
            return;
        }

        receivingPattern = false;

        applyITO();

        Serial.println("ITO_APPLIED");

        return;
    }


    // --------------------------------------------------------
    // PATTERN ROW
    // --------------------------------------------------------

    if (receivingPattern)
    {
        if (currentRow >= ROWS)
        {
            Serial.println("ERROR TOO_MANY_ROWS");
            return;
        }

        if (line.length() != COLS)
        {
            Serial.println("ERROR ROW_LENGTH");
            receivingPattern = false;
            return;
        }

        for (int c = 0; c < COLS; c++)
        {
            char value = line.charAt(c);

            if (value == '0')
            {
                pattern[currentRow][c] = 0;
            }
            else if (value == '1')
            {
                pattern[currentRow][c] = 1;
            }
            else
            {
                Serial.println("ERROR INVALID_BIT");
                receivingPattern = false;
                return;
            }
        }

        currentRow++;

        return;
    }

    Serial.println("ERROR UNKNOWN_COMMAND");
}


// ------------------------------------------------------------
// APPLY COMPLETE PATTERN
// ------------------------------------------------------------

void applyITO()
{
    /*
      This is deliberately separated from the serial receiver.

      At this point:

          pattern[r][c]

      contains the entire desired electrode configuration.

      The actual implementation depends on your ITO driver.
    */

    for (int r = 0; r < ROWS; r++)
    {
        for (int c = 0; c < COLS; c++)
        {
            setITOCell(
                r,
                c,
                pattern[r][c]
            );
        }
    }
}


// ------------------------------------------------------------
// HARDWARE INITIALISATION
// ------------------------------------------------------------

void initialiseITOHardware()
{
    /*
      Put your actual driver initialisation here.

      Example:

          SPI.begin();

      or configure GPIO expanders / multiplexers / DACs.

      This example intentionally does not assume a particular
      electrical topology.
    */
}


// ------------------------------------------------------------
// ITO CELL DRIVER
// ------------------------------------------------------------

void setITOCell(
    int row,
    int col,
    uint8_t state
)
{
    /*
      HARDWARE-SPECIFIC SECTION.

      state == 0
          electrode OFF

      state == 1
          electrode ON

      Replace this with the actual circuit controlling your
      ITO electrodes.

      For example, if you use external row/column drivers:

          selectRow(row);
          selectColumn(col);
          setVoltage(state);

      Do NOT simply use:

          digitalWrite(row, state);

      unless your actual hardware has been designed so that
      this is electrically appropriate.
    */

    // Placeholder.
    //
    // Your real ITO driver goes here.
}


// ------------------------------------------------------------
// OPTIONAL CLEAR
// ------------------------------------------------------------

void clearITO()
{
    for (int r = 0; r < ROWS; r++)
    {
        for (int c = 0; c < COLS; c++)
        {
            pattern[r][c] = 0;

            setITOCell(
                r,
                c,
                0
            );
        }
    }
}
