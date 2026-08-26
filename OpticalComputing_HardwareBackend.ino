/*
    1-AXIS ITO OPTICAL BENCH CONTROLLER

    Python sends:

        BEGIN 32
        01010101010101010101010101010101
        END

    or, for another axis length:

        BEGIN 128
        010101...
        END

    The Arduino stores the complete 1D pattern and then
    applies it to the external ITO/light-valve electronics.

    IMPORTANT:
    Do NOT connect an ITO electrode directly to Arduino GPIO
    unless the electrical design explicitly permits it.

    Replace setITOCell() with the actual external driver.
*/

#include <Arduino.h>
#include <string.h>

#define AXIS_LENGTH 32

uint8_t pattern[AXIS_LENGTH];

bool receivingPattern = false;

int currentLength = 0;


// ============================================================
// SETUP
// ============================================================

void setup()
{
    Serial.begin(115200);

    memset(
        pattern,
        0,
        sizeof(pattern)
    );

    initialiseITOHardware();

    Serial.println("ITO_READY");
}


// ============================================================
// MAIN LOOP
// ============================================================

void loop()
{
    if (Serial.available())
    {
        String line =
            Serial.readStringUntil('\n');

        line.trim();

        if (line.length() == 0)
            return;

        processLine(line);
    }
}


// ============================================================
// SERIAL PROTOCOL
// ============================================================

void processLine(String &line)
{
    // --------------------------------------------------------
    // BEGIN
    // --------------------------------------------------------

    if (line.startsWith("BEGIN"))
    {
        int space =
            line.indexOf(' ');

        if (space < 0)
        {
            Serial.println("ERROR BEGIN");
            return;
        }

        int requestedLength =
            line.substring(
                space + 1
            ).toInt();

        if (
            requestedLength !=
            AXIS_LENGTH
        )
        {
            Serial.println(
                "ERROR DIMENSIONS"
            );

            receivingPattern = false;

            return;
        }

        currentLength = 0;

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
            Serial.println(
                "ERROR NO_PATTERN"
            );

            return;
        }

        if (
            currentLength !=
            AXIS_LENGTH
        )
        {
            Serial.println(
                "ERROR LENGTH"
            );

            receivingPattern = false;

            return;
        }

        receivingPattern = false;

        applyITO();

        Serial.println(
            "ITO_APPLIED"
        );

        return;
    }


    // --------------------------------------------------------
    // 1D PATTERN
    // --------------------------------------------------------

    if (receivingPattern)
    {
        if (
            line.length() !=
            AXIS_LENGTH
        )
        {
            Serial.println(
                "ERROR ROW_LENGTH"
            );

            receivingPattern = false;

            return;
        }

        for (
            int i = 0;
            i < AXIS_LENGTH;
            i++
        )
        {
            char value =
                line.charAt(i);

            if (value == '0')
            {
                pattern[i] = 0;
            }
            else if (value == '1')
            {
                pattern[i] = 1;
            }
            else
            {
                Serial.println(
                    "ERROR INVALID_BIT"
                );

                receivingPattern = false;

                return;
            }
        }

        currentLength =
            AXIS_LENGTH;

        return;
    }

    Serial.println(
        "ERROR UNKNOWN_COMMAND"
    );
}


// ============================================================
// APPLY ITO
// ============================================================

void applyITO()
{
    for (
        int i = 0;
        i < AXIS_LENGTH;
        i++
    )
    {
        setITOCell(
            i,
            pattern[i]
        );
    }
}


// ============================================================
// HARDWARE INITIALISATION
// ============================================================

void initialiseITOHardware()
{
    /*
        Put the actual external ITO driver
        initialization here.

        Examples:

            SPI.begin();

            shift-register initialization

            GPIO expander initialization

            analogue multiplexer initialization

            DAC initialization
    */
}


// ============================================================
// ONE AXIS ITO DRIVER
// ============================================================

void setITOCell(
    int position,
    uint8_t state
)
{
    /*
        position:
            0 ... AXIS_LENGTH-1

        state:
            0 = OFF
            1 = ON

        Replace this function with the electronics
        actually driving your ITO axis.

        Example conceptual interface:

            selectITOPosition(position);
            setITOState(state);

        Do not assume Arduino GPIO voltage/current is
        appropriate for the physical ITO device.
    */
}


// ============================================================
// CLEAR
// ============================================================

void clearITO()
{
    for (
        int i = 0;
        i < AXIS_LENGTH;
        i++
    )
    {
        pattern[i] = 0;

        setITOCell(
            i,
            0
        );
    }
}
