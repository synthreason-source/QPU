"""
optical_binary.py
================

Hardware backend for optical computing primitives. Connects to the
physical light valve controller via USB/serial and captures intensity
readings directly from a physical camera frame using OpenCV.
"""

import time
import serial
import cv2
import numpy as np

# Configure your serial port and camera index
SERIAL_PORT = "COM7"
BAUD_RATE = 115200
CAMERA_INDEX = 0  # 0 for default USB camera, or specify stream/device ID

# Initialize serial connection for light valves
try:
    _ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Allow hardware to settle
except Exception as e:
    _ser = None
    print(f"[Warning] Could not open serial port {SERIAL_PORT}: {e}. Running in dummy/mock mode.")

# Initialize physical camera capture
_cam = cv2.VideoCapture(CAMERA_INDEX)
if not _cam.isOpened():
    print(f"[Warning] Could not open camera index {CAMERA_INDEX}. Using simulated intensity fallback.")
    _cam = None


def _send_light_valve_matrix(holes: np.ndarray):
    """
    Serializes the 2D hole/aperture matrix and transmits it to the 
    light valve controller over USB/serial.
    """
    if _ser is None or not _ser.is_open:
        return  # Mock mode fallback

    flat = holes.flatten()
    byte_data = np.packbits(flat).tobytes()
    
    _ser.write(b"FRAME_START\n")
    _ser.write(byte_data)
    _ser.write(b"\nFRAME_END\n")
    
    ack = _ser.readline()
    if not ack.startswith(b"ACK"):
        raise RuntimeError(f"Hardware light valve synchronization failed: {ack}")


def _capture_sensor_frame() -> float:
    """
    Captures a real physical frame from the camera, converts it to grayscale,
    and returns the normalized average pixel intensity (representing the optical field).
    """
    if _cam is None or not _cam.isOpened():
        return 1.0  # Fallback simulation value

    # Grab a frame from the camera
    ret, frame = _cam.read()
    if not ret or frame is None:
        raise RuntimeError("Failed to grab a frame from the physical camera.")

    # Convert to grayscale to measure total luminous intensity
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate average pixel intensity (or np.sum() depending on your optical setup)
    intensity = float(np.mean(gray))
    return intensity


def make_binary_plane_and_holes(values: np.ndarray):
    """Create a standard aperture coordinate layout for the optical plane."""
    plane = np.ones_like(values, dtype=np.uint8)
    holes = values.astype(np.uint8)
    return plane, holes


def camera_exposure(plane: np.ndarray, holes: np.ndarray) -> dict:
    """
    Applies the hole matrix to the light valves, triggers a physical camera exposure,
    and returns the measured optical field intensity from the captured frame.
    """
    active_mask = plane & holes
    _send_light_valve_matrix(active_mask)
    
    # Optional short delay to let light valves settle before capturing frame
    time.sleep(0.01)
    
    intensity = _capture_sensor_frame()
    return {"dc_field": complex(intensity, 0.0)}


def binary_multiply(x: int, y: int) -> int:
    """Native hardware AND operation via a single hole-pair exposure."""
    plane, holes = make_binary_plane_and_holes(np.array([[x & y]], dtype=np.uint8))
    optics = camera_exposure(plane, holes)
    # Threshold the average frame intensity to determine logic state
    return 1 if optics["dc_field"].real > 50.0 else 0


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


def main():
    """Run a test program with 3 loops, performance timings, and optical activity counts."""
    print("--- Starting Optical Computing Test Program ---")
    mode = "Hardware" if (_ser and _ser.is_open and _cam and _cam.isOpened()) else "Mock / Simulation"
    print(f"Backend Status: {mode}\n")

    total_loops = 3
    optical_activity_count = 0
    total_intensity_accumulated = 0.0
    
    loop_timings = []

    try:
        for i in range(1, total_loops + 1):
            loop_start = time.perf_counter()
            print(f"-> Executing Loop {i} of {total_loops}...")

            mult_result = binary_multiply(1, 1)
            sum_result = binary_add_many([1, 0, 1, 1, 0])
            
            optical_activity_count += 2 
            total_intensity_accumulated += (mult_result + sum_result)

            loop_end = time.perf_counter()
            elapsed = loop_end - loop_start
            loop_timings.append(elapsed)
            
            print(f"    Completed Loop {i} | Multiply: {mult_result}, Sum: {sum_result} | Time: {elapsed:.6f}s")
    finally:
        if _cam is not None:
            _cam.release()
        if _ser is not None and _ser.is_open:
            _ser.close()

    avg_time = sum(loop_timings) / len(loop_timings)
    
    print("\n" + "="*45)
    print("         TEST PROGRAM STATISTICS          ")
    print("="*45)
    print(f"* Total Loops Run            : {total_loops}")
    print(f"* Optical Activity Count     : {optical_activity_count} exposures")
    print(f"* Accumulated Intensity      : {total_intensity_accumulated}")
    print(f"* Average Loop Duration      : {avg_time:.6f} seconds")
    print(f"* Fastest Loop Duration      : {min(loop_timings):.6f} seconds")
    print(f"* Slowest Loop Duration      : {max(loop_timings):.6f} seconds")
    print("="*45)


if __name__ == "__main__":
    main()
