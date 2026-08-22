"""
optical_hardware_interface.py
==============================
Abstraction layer for an SLM + camera optical vector-matrix multiplier
(the classic Goodman & Athale 1978 incoherent architecture: a matrix is
displayed as pixel transmittance on an SLM, a vector is encoded as
per-row light-source intensities, and a summing lens/lenslet array
focuses each column's total transmitted power onto one camera pixel -
giving sum_k A[i,k]*B[k,j] directly from measured intensity).

Two kinds of classes are provided:

  * MockSLM / MockCamera / MockOpticalBench
        A physically-modeled *software* stand-in: real linear-algebra
        physics (intensity = illumination @ transmittance), plus
        realistic shot noise, read noise, and ADC quantization. This
        is fully runnable and tested right now, with no hardware.

  * MeadowlarkSLM, ThorlabsCamera, OpenCVCamera
        Skeletons/wrappers for real devices.
        - MeadowlarkSLM / ThorlabsCamera are UNTESTED STUBS. I have no
          physical SLM, no vendor SDK installed, and no way to verify
          these calls in this environment - treat them as a starting
          point to fill in against your exact SDK version and model,
          not as working code.
        - OpenCVCamera is a real, plausible-to-test implementation for
          any generic UVC/USB webcam, since it only needs the
          `opencv-python` package (`pip install opencv-python`), no
          vendor SDK. You should still confirm exposure/gain settings
          match your actual sensor.

IMPORTANT: nothing in this file has been run against real hardware.
Treat all hardware-facing code as a draft to validate on your bench,
not a guarantee.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


# ---------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------

class SLM(ABC):
    """A device that displays a 2D pattern of transmittance/phase values."""

    @abstractmethod
    def get_shape(self) -> Tuple[int, int]:
        """(n_rows, n_cols) of the SLM's addressable pattern."""

    @abstractmethod
    def set_pattern(self, pattern: np.ndarray) -> None:
        """pattern: float array in [0, 1], shape == get_shape()."""

    def close(self) -> None:
        pass


class Camera(ABC):
    """A device that captures an intensity image."""

    @abstractmethod
    def get_shape(self) -> Tuple[int, int]:
        """(height, width) of captured frames."""

    @abstractmethod
    def capture(self) -> np.ndarray:
        """Returns a float array of raw intensity counts, shape == get_shape()."""

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------
# Mock backend: real physics, no hardware
# ---------------------------------------------------------------------

class MockOpticalBench:
    """
    Shared physical state for a mock SLM + camera pair.

    Model: illumination is a length-n_rows vector of source intensities
    (one "lane" of light per row). The SLM mask sets per-cell
    transmittance. A summing lens integrates transmitted power down
    each column, so the ideal (noiseless) camera readout is:

        ideal[j] = sum_i illumination[i] * mask[i, j]

    which is exactly one row of a matrix-vector product. Realistic
    imperfections are then layered on top:
        - Poisson-like shot noise (scales with sqrt of signal)
        - Gaussian read noise (camera electronics floor)
        - finite ADC bit depth (quantization)
    """

    def __init__(self, n_rows: int, n_cols: int, gain: float = 200.0,
                 read_noise_std: float = 2.0, shot_noise_scale: float = 1.0,
                 adc_bits: int = 12, seed: int = 0):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.gain = gain                      # ADC counts per unit of ideal signal
        self.read_noise_std = read_noise_std  # in ADC counts
        self.shot_noise_scale = shot_noise_scale
        self.adc_max = 2 ** adc_bits - 1
        self.mask = np.zeros((n_rows, n_cols))
        self.illumination = np.zeros(n_rows)
        self.rng = np.random.default_rng(seed)

    def set_pattern(self, pattern: np.ndarray) -> None:
        assert pattern.shape == (self.n_rows, self.n_cols)
        self.mask = np.clip(pattern, 0.0, 1.0)

    def set_illumination(self, vec: np.ndarray) -> None:
        assert vec.shape == (self.n_rows,)
        self.illumination = np.clip(vec, 0.0, None)

    def capture(self) -> np.ndarray:
        ideal = self.illumination @ self.mask           # shape (n_cols,)
        signal = ideal * self.gain
        shot = self.rng.normal(0.0, np.sqrt(np.maximum(signal, 1e-9)) * self.shot_noise_scale)
        read = self.rng.normal(0.0, self.read_noise_std, size=signal.shape)
        noisy = signal + shot + read
        return np.clip(np.round(noisy), 0, self.adc_max)


class MockSLM(SLM):
    def __init__(self, bench: MockOpticalBench):
        self.bench = bench

    def get_shape(self) -> Tuple[int, int]:
        return (self.bench.n_rows, self.bench.n_cols)

    def set_pattern(self, pattern: np.ndarray) -> None:
        self.bench.set_pattern(pattern)


class MockCamera(Camera):
    def __init__(self, bench: MockOpticalBench):
        self.bench = bench

    def get_shape(self) -> Tuple[int, int]:
        return (1, self.bench.n_cols)

    def capture(self) -> np.ndarray:
        return self.bench.capture().reshape(1, -1)  # match get_shape()'s (1, n_cols)


# ---------------------------------------------------------------------
# Real hardware: UNTESTED skeletons - adapt to your exact SDK/model
# ---------------------------------------------------------------------

class MeadowlarkSLM(SLM):
    """
    Skeleton for a Meadowlark Optics 'Blink' SLM, normally controlled
    via their C SDK (Blink_C_wrapper.dll) through ctypes.

    UNTESTED: I have no physical SLM or the vendor SDK installed here,
    so none of the ctypes calls below have been run. Use Meadowlark's
    own `Blink_C_wrapper` Python example (ships with the SDK) as the
    source of truth for exact function names/signatures for your
    firmware version, and treat this class as a shape to fill in.
    """

    def __init__(self, dll_path: str, width: int, height: int, bit_depth: int = 8):
        import ctypes  # local import: only needed if you actually use this class
        self._width, self._height = width, height
        self._bit_depth = bit_depth
        self.slm_lib = ctypes.cdll.LoadLibrary(dll_path)
        # TODO: call the SDK's constructor/init routine, e.g. something like:
        #   self.slm_lib.Create_SDK()
        #   num_boards_found = ctypes.c_uint(0)
        #   constructed_okay = ctypes.c_uint(-1)
        #   self.slm_lib.Create_SDK(ctypes.byref(num_boards_found), ctypes.byref(constructed_okay), ...)
        # Exact args differ by SDK version - check your installed headers/examples.
        raise NotImplementedError(
            "Fill in the exact Blink SDK initialization calls for your "
            "SLM model/firmware before using this class."
        )

    def get_shape(self) -> Tuple[int, int]:
        return (self._height, self._width)

    def set_pattern(self, pattern: np.ndarray) -> None:
        # TODO: convert `pattern` (floats in [0,1]) to the SDK's expected
        # 8/16-bit grayscale buffer, e.g.:
        #   img = (pattern * (2**self._bit_depth - 1)).astype(np.uint8 or np.uint16)
        #   c_img = img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        #   self.slm_lib.Write_image(board_number, c_img, ...)
        raise NotImplementedError

    def close(self) -> None:
        # TODO: self.slm_lib.Delete_SDK()
        pass


class ThorlabsCamera(Camera):
    """
    Skeleton for a Thorlabs scientific camera via `thorlabs_tsi_sdk`
    (pip install thorlabs_tsi_sdk; also requires the Thorlabs TSI SDK
    DLLs to be installed/on PATH).

    UNTESTED: no physical camera or SDK available here to verify
    against. Cross-check against Thorlabs' own example scripts that
    ship with the SDK for your exact camera model.
    """

    def __init__(self, exposure_us: int = 10000):
        from thorlabs_tsi_sdk.tl_camera import TLCameraSDK  # TODO: verify import path/version
        self._sdk = TLCameraSDK()
        cams = self._sdk.discover_available_cameras()
        if not cams:
            raise RuntimeError("No Thorlabs camera found")
        self._cam = self._sdk.open_camera(cams[0])
        self._cam.exposure_time_us = exposure_us
        self._cam.arm(2)
        self._cam.issue_software_trigger()

    def get_shape(self) -> Tuple[int, int]:
        return (self._cam.image_height_pixels, self._cam.image_width_pixels)

    def capture(self) -> np.ndarray:
        frame = self._cam.get_pending_frame_or_null()
        if frame is None:
            raise RuntimeError("Frame grab timed out")
        return np.array(frame.image_buffer, dtype=float)

    def close(self) -> None:
        self._cam.disarm()
        self._cam.dispose()
        self._sdk.dispose()


class OpenCVCamera(Camera):
    """
    Real, testable implementation for any generic UVC/USB webcam via
    OpenCV. Only needs `pip install opencv-python` - no vendor SDK.
    Still confirm your camera's exposure/gain/gamma settings are
    appropriate for quantitative intensity measurement (many webcams
    apply auto-exposure and gamma correction by default, which will
    corrupt a linear intensity readout unless disabled).
    """

    def __init__(self, index: int = 0, disable_auto_exposure: bool = True):
        import cv2
        self._cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")
        if disable_auto_exposure:
            # Value/semantics of this flag are backend-dependent (V4L2 vs DirectShow);
            # verify it actually disables auto-exposure on your camera/driver.
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

    def get_shape(self) -> Tuple[int, int]:
        w = int(self.cap.get(self._cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(self._cv2.CAP_PROP_FRAME_HEIGHT))
        return (h, w)

    def capture(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Frame grab failed")
        gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        return gray.astype(float)

    def close(self) -> None:
        self.cap.release()
