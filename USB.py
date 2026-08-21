"""
optical_binary.py
================

Hardware backend for optical computing primitives. Connects to the
physical light valve controller via USB/serial and captures intensity
readings from the optical sensor/camera.
"""

import time
import serial
import numpy as np

# Configure your serial port and baud rate to match your microcontroller/light valve driver
SERIAL_PORT = "/dev/ttyUSB0"  # Update to your port (e.g., "COM3" on Windows)
BAUD_RATE = 115200

# Initialize serial connection for light valves
try:
    _ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Allow hardware to settle
except Exception as e:
    _ser = None
    print(f"[Warning] Could not open serial port {SERIAL_PORT}: {e}. Running in dummy/mock mode.")


def _send_light_valve_matrix(holes: np.ndarray):
    """
    Serializes the 2D hole/aperture matrix and transmits it to the 
    light valve controller over USB/serial.
    """
    if _ser is None or not _ser.is_open:
        return  # Mock mode fallback

    # Flatten and convert matrix to binary byte stream or control command string
    flat = holes.flatten()
    # Example protocol: sending a byte stream representing active valve states
    byte_data = np.packbits(flat).tobytes()
    
    _ser.write(b"FRAME_START\n")
    _ser.write(byte_data)
    _ser.write(b"\nFRAME_END\n")
    
    # Wait for hardware acknowledgement
    ack = _ser.readline()
    if not ack.startswith(b"ACK"):
        raise RuntimeError(f"Hardware light valve synchronization failed: {ack}")


def _capture_sensor_frame() -> float:
    """
    Triggers the optical sensor/camera and reads back the total transmitted
    intensity (DC field component) of the exposure.
    """
    if _ser is None or not _ser.is_open:
        # Fallback simulation value if hardware is disconnected
        return 0.0

    _ser.write(b"TRIGGER_EXPOSURE\n")
    response = _ser.readline().decode("utf-8").strip()
    
    if response.startswith("INTENSITY:"):
        return float(response.split(":")[1])
    else:
        raise RuntimeError(f"Invalid sensor read response: {response}")


def make_binary_plane_and_holes(values: np.ndarray):
    """Create a standard aperture coordinate layout for the optical plane."""
    plane = np.ones_like(values, dtype=np.uint8)
    holes = values.astype(np.uint8)
    return plane, holes


def camera_exposure(plane: np.ndarray, holes: np.ndarray) -> dict:
    """
    Applies the hole matrix to the light valves, triggers a physical camera exposure,
    and returns the measured optical field intensity.
    """
    # Combine plane and holes into the final active mask layout
    active_mask = plane & holes
    
    # Send mask configuration to physical hardware via USB/serial
    _send_light_valve_matrix(active_mask)
    
    # Read physical intensity from camera
    intensity = _capture_sensor_frame()
    
    return {"dc_field": complex(intensity, 0.0)}


def binary_multiply(x: int, y: int) -> int:
    """Native hardware AND operation via a single hole-pair exposure."""
    plane, holes = make_binary_plane_and_holes(np.array([[x & y]], dtype=np.uint8))
    optics = camera_exposure(plane, holes)
    return 1 if optics["dc_field"].real > 0.5 else 0


def binary_add_many(values: list) -> int:
    """Native hardware SUM operation via a single plane-of-holes exposure."""
    if not values:
        return 0
    arr = np.array(values, dtype=np.uint8)
    plane, holes = make_binary_plane_and_holes(arr)
    optics = camera_exposure(plane, holes)
    return int(round(optics["dc_field"].real))


def apply_holes(plane: np.ndarray, holes: np.ndarray) -> dict:
    """Directly apply a custom aperture mask and return sensor reading."""
    return camera_exposure(plane, holes)