#!/usr/bin/env python3

"""
32-MODE AREA-INTENSITY OPTICAL BENCH
====================================

Optical model:

        32 correlated optical modes
                    |
                    v
              optical field
                    |
                    v
             1-axis ITO
                    |
                    v
             focusing lens
                    |
                    v
                 camera
                    |
                    v
             camera image
                    |
                    v
        area-integrated intensity


Primary observable:

        I_area(t) = sum_{x,y in ROI} |E(x,y,t)|^2

This program supports:

    simulate
    hardware
    compare

Arduino protocol:

    BEGIN 32
    010101010101...
    END

The Arduino ITO axis length must match AXIS_LENGTH below.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

AXIS_LENGTH = 32

NUM_MODES = 32

EPS = 1e-12


# ============================================================
# OPTICAL CONFIGURATION
# ============================================================

@dataclass
class OpticalConfig:

    # --------------------------------------------------------
    # Optical modes
    # --------------------------------------------------------

    num_modes: int = 32

    # --------------------------------------------------------
    # ITO axis
    # --------------------------------------------------------

    axis_length: int = AXIS_LENGTH

    # --------------------------------------------------------
    # Simulated optical plane
    # --------------------------------------------------------

    optical_pixels: int = 256

    # --------------------------------------------------------
    # Beam
    # --------------------------------------------------------

    beam_sigma: float = 0.42

    # --------------------------------------------------------
    # Mode correlations
    # --------------------------------------------------------

    amplitude_correlation: float = 0.75

    phase_correlation: float = 0.90

    # --------------------------------------------------------
    # Noise
    # --------------------------------------------------------

    field_noise: float = 0.002

    camera_noise: float = 0.001

    # --------------------------------------------------------
    # Lens
    # --------------------------------------------------------

    lens_strength: float = 1.0

    # --------------------------------------------------------
    # ITO threshold
    # --------------------------------------------------------

    ito_threshold: float = 0.50

    # --------------------------------------------------------
    # Camera ROI
    #
    # None = entire camera
    #
    # Or specify:
    #
    # roi_x0
    # roi_x1
    # roi_y0
    # roi_y1
    # --------------------------------------------------------

    roi_x0: int = 64
    roi_x1: int = 192

    roi_y0: int = 64
    roi_y1: int = 192


# ============================================================
# CORRELATED 32-MODE SOURCE
# ============================================================

class Correlated32ModeLaser:

    """
    Generates a 32-mode complex optical field.

    The modes share correlated amplitude and phase components.

    This is a mathematical correlated-field model.

    It does NOT by itself demonstrate physical quantum
    entanglement.
    """

    def __init__(
        self,
        config: OpticalConfig,
        seed: int | None = None
    ):

        self.cfg = config

        self.rng = np.random.default_rng(
            seed
        )

        self.common_phase = (
            self.rng.uniform(
                -np.pi,
                np.pi
            )
        )

        self.common_amplitude = 1.0

        self.mode_amplitudes = (
            np.ones(
                config.num_modes
            )
        )

        self.mode_phases = (
            np.zeros(
                config.num_modes
            )
        )

        self._initialize()


    # --------------------------------------------------------
    # INITIALIZE
    # --------------------------------------------------------

    def _initialize(self):

        independent_amp = (
            self.rng.normal(
                0.0,
                0.08,
                self.cfg.num_modes
            )
        )

        self.mode_amplitudes = (
            self.common_amplitude
            *
            (
                1.0
                +
                self.cfg.amplitude_correlation
                * independent_amp
            )
        )

        independent_phase = (
            self.rng.normal(
                0.0,
                0.25,
                self.cfg.num_modes
            )
        )

        self.mode_phases = (
            self.cfg.phase_correlation
            *
            self.common_phase
            +
            (
                1.0
                -
                self.cfg.phase_correlation
            )
            *
            independent_phase
        )


    # --------------------------------------------------------
    # EVOLVE SOURCE
    # --------------------------------------------------------

    def step(
        self,
        control_phase=0.0
    ):

        self.common_phase += (
            control_phase
        )

        independent_phase = (
            self.rng.normal(
                0.0,
                0.10,
                self.cfg.num_modes
            )
        )

        self.mode_phases = (
            self.cfg.phase_correlation
            *
            self.common_phase
            +
            (
                1.0
                -
                self.cfg.phase_correlation
            )
            *
            independent_phase
        )

        amplitude_noise = (
            self.rng.normal(
                0.0,
                0.015,
                self.cfg.num_modes
            )
        )

        self.mode_amplitudes *= (
            1.0
            +
            amplitude_noise
        )

        self.mode_amplitudes = np.maximum(
            self.mode_amplitudes,
            0.0
        )


    # --------------------------------------------------------
    # COMPLEX MODE COEFFICIENTS
    # --------------------------------------------------------

    def coefficients(self):

        return (
            self.mode_amplitudes
            *
            np.exp(
                1j * self.mode_phases
            )
        )


# ============================================================
# 32 SPATIAL MODE BASIS
# ============================================================

class SpatialModeBasis:

    def __init__(
        self,
        config: OpticalConfig,
        seed: int | None = None
    ):

        self.cfg = config

        rng = np.random.default_rng(
            seed
        )

        N = config.optical_pixels

        coordinate = np.linspace(
            -1.0,
            1.0,
            N
        )

        self.X, self.Y = np.meshgrid(
            coordinate,
            coordinate
        )

        radius2 = (
            self.X ** 2
            +
            self.Y ** 2
        )

        self.envelope = np.exp(
            -radius2
            /
            (
                2.0
                *
                config.beam_sigma ** 2
            )
        )

        self.modes = []

        for k in range(
            config.num_modes
        ):

            angle = (
                2.0
                *
                np.pi
                *
                k
                /
                config.num_modes
            )

            spatial_frequency = (
                1.0
                +
                (k % 8)
                *
                0.55
            )

            fx = (
                spatial_frequency
                *
                np.cos(angle)
            )

            fy = (
                spatial_frequency
                *
                np.sin(angle)
            )

            phase = (
                2.0
                *
                np.pi
                *
                (
                    fx * self.X
                    +
                    fy * self.Y
                )
            )

            phase += rng.uniform(
                -np.pi,
                np.pi
            )

            mode = (
                self.envelope
                *
                np.exp(
                    1j * phase
                )
            )

            norm = np.sqrt(
                np.sum(
                    np.abs(mode) ** 2
                )
            )

            if norm > EPS:
                mode /= norm

            self.modes.append(
                mode
            )

        self.modes = np.asarray(
            self.modes
        )


    # --------------------------------------------------------
    # SUPERPOSE
    # --------------------------------------------------------

    def superpose(
        self,
        coefficients
    ):

        field = np.zeros(
            (
                self.cfg.optical_pixels,
                self.cfg.optical_pixels
            ),
            dtype=np.complex128
        )

        for k in range(
            self.cfg.num_modes
        ):

            field += (
                coefficients[k]
                *
                self.modes[k]
            )

        return field


# ============================================================
# 1D ITO AXIS
# ============================================================

class ITOAxis:

    def __init__(
        self,
        config: OpticalConfig
    ):

        self.cfg = config

        self.pattern = np.zeros(
            config.axis_length,
            dtype=np.uint8
        )


    # --------------------------------------------------------
    # GENERATE FROM FIELD
    # --------------------------------------------------------

    def pattern_from_field(
        self,
        field
    ):

        """
        Project the 2D optical field onto the
        single ITO axis.

        Each axis position receives the average
        optical intensity associated with that
        longitudinal position.
        """

        intensity = (
            np.abs(field) ** 2
        )

        N = self.cfg.optical_pixels

        axis = self.cfg.axis_length

        values = np.zeros(
            axis,
            dtype=np.float64
        )

        # Divide the optical plane into axis bins.

        for i in range(axis):

            x0 = int(
                i * N / axis
            )

            x1 = int(
                (i + 1)
                * N
                / axis
            )

            x0 = max(
                0,
                min(
                    N - 1,
                    x0
                )
            )

            x1 = max(
                x0 + 1,
                min(
                    N,
                    x1
                )
            )

            values[i] = np.mean(
                intensity[
                    :,
                    x0:x1
                ]
            )

        maximum = np.max(
            values
        )

        if maximum > EPS:

            normalized = (
                values
                /
                maximum
            )

        else:

            normalized = values

        self.pattern = (
            normalized
            >=
            self.cfg.ito_threshold
        ).astype(
            np.uint8
        )

        return self.pattern


    # --------------------------------------------------------
    # PHASE PATTERN
    # --------------------------------------------------------

    def pattern_from_phase(
        self,
        field
    ):

        phase = np.angle(
            field
        )

        axis_phase = np.mean(
            phase,
            axis=0
        )

        N = len(
            axis_phase
        )

        axis = self.cfg.axis_length

        values = np.zeros(
            axis
        )

        for i in range(axis):

            x0 = int(
                i * N / axis
            )

            x1 = int(
                (i + 1)
                * N
                / axis
            )

            values[i] = np.mean(
                np.cos(
                    axis_phase[x0:x1]
                )
            )

        self.pattern = (
            values > 0
        ).astype(
            np.uint8
        )

        return self.pattern


    # --------------------------------------------------------
    # RANDOM
    # --------------------------------------------------------

    def random(
        self,
        probability=0.5,
        rng=None
    ):

        if rng is None:

            rng = np.random.default_rng()

        self.pattern = (
            rng.random(
                self.cfg.axis_length
            )
            <
            probability
        ).astype(
            np.uint8
        )

        return self.pattern


# ============================================================
# APPLY 1D ITO TO 2D FIELD
# ============================================================

class ITOProjection:

    @staticmethod
    def expand_axis(
        pattern,
        width
    ):

        """
        Expand a 1D ITO pattern across the
        full 2D optical plane.
        """

        axis_length = len(
            pattern
        )

        result = np.zeros(
            (
                width,
                width
            ),
            dtype=np.float64
        )

        for i in range(
            axis_length
        ):

            x0 = int(
                i * width
                /
                axis_length
            )

            x1 = int(
                (i + 1)
                * width
                /
                axis_length
            )

            result[
                :,
                x0:x1
            ] = pattern[i]

        return result


    @staticmethod
    def apply(
        field,
        pattern
    ):

        valve = (
            ITOProjection.expand_axis(
                pattern,
                field.shape[0]
            )
        )

        return (
            field
            *
            valve
        )


# ============================================================
# FOCUSING OPTICS
# ============================================================

class FocusingOptics:

    def __init__(
        self,
        config: OpticalConfig
    ):

        self.cfg = config


    # --------------------------------------------------------
    # LENS
    # --------------------------------------------------------

    def apply_lens(
        self,
        field
    ):

        N = field.shape[0]

        x = np.linspace(
            -1.0,
            1.0,
            N
        )

        y = np.linspace(
            -1.0,
            1.0,
            N
        )

        X, Y = np.meshgrid(
            x,
            y
        )

        r2 = (
            X ** 2
            +
            Y ** 2
        )

        phase = (
            -
            np.pi
            *
            self.cfg.lens_strength
            *
            r2
        )

        return (
            field
            *
            np.exp(
                1j * phase
            )
        )


    # --------------------------------------------------------
    # FOCUS
    # --------------------------------------------------------

    def focus(
        self,
        field
    ):

        field = (
            self.apply_lens(
                field
            )
        )

        focal_field = np.fft.fftshift(
            np.fft.fft2(
                np.fft.ifftshift(
                    field
                )
            )
        )

        maximum = np.max(
            np.abs(
                focal_field
            )
        )

        if maximum > EPS:

            focal_field /= maximum

        return focal_field


    # --------------------------------------------------------
    # INTENSITY
    # --------------------------------------------------------

    def intensity(
        self,
        field
    ):

        focal_field = (
            self.focus(
                field
            )
        )

        return (
            np.abs(
                focal_field
            ) ** 2
        )


# ============================================================
# AREA INTEGRATION
# ============================================================

class AreaDetector:

    def __init__(
        self,
        config: OpticalConfig
    ):

        self.cfg = config

        self.history = []


    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    def roi(
        self,
        image
    ):

        y0 = self.cfg.roi_y0
        y1 = self.cfg.roi_y1

        x0 = self.cfg.roi_x0
        x1 = self.cfg.roi_x1

        return image[
            y0:y1,
            x0:x1
        ]


    # --------------------------------------------------------
    # AREA INTENSITY
    # --------------------------------------------------------

    def area_intensity(
        self,
        image
    ):

        region = self.roi(
            image
        )

        value = float(
            np.sum(region)
        )

        self.history.append(
            value
        )

        return value


    # --------------------------------------------------------
    # AREA MEAN
    # --------------------------------------------------------

    def area_mean(
        self,
        image
    ):

        region = self.roi(
            image
        )

        return float(
            np.mean(region)
        )


    # --------------------------------------------------------
    # AREA VARIANCE
    # --------------------------------------------------------

    def area_variance(
        self,
        image
    ):

        region = self.roi(
            image
        )

        return float(
            np.var(region)
        )


    # --------------------------------------------------------
    # AREA CONTRAST
    # --------------------------------------------------------

    def area_contrast(
        self,
        image
    ):

        region = self.roi(
            image
        )

        mean = np.mean(
            region
        )

        if mean < EPS:

            return 0.0

        return float(
            np.std(region)
            /
            mean
        )


    # --------------------------------------------------------
    # TIME SERIES
    # --------------------------------------------------------

    def series(self):

        return np.asarray(
            self.history,
            dtype=np.float64
        )


# ============================================================
# SIMULATED CAMERA
# ============================================================

class SimulatedCamera:

    def __init__(
        self,
        config: OpticalConfig,
        seed=None
    ):

        self.cfg = config

        self.rng = np.random.default_rng(
            seed
        )


    def capture(
        self,
        image
    ):

        if self.cfg.camera_noise > 0:

            noise = (
                self.cfg.camera_noise
                *
                self.rng.normal(
                    size=image.shape
                )
            )

            image = (
                image
                +
                noise
            )

        return np.maximum(
            image,
            0.0
        )


# ============================================================
# REAL CAMERA
# ============================================================

class RealCamera:

    def __init__(
        self,
        camera_index=0
    ):

        self.camera_index = (
            camera_index
        )

        self.cap = None


    def open(self):

        import cv2

        self.cap = cv2.VideoCapture(
            self.camera_index
        )

        if not self.cap.isOpened():

            raise RuntimeError(
                "Unable to open camera "
                f"{self.camera_index}"
            )


    def capture(self):

        import cv2

        ok, frame = (
            self.cap.read()
        )

        if not ok:

            raise RuntimeError(
                "Camera acquisition failed"
            )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        return gray.astype(
            np.float64
        )


    def close(self):

        if self.cap is not None:

            self.cap.release()

            self.cap = None


# ============================================================
# ARDUINO DRIVER
# ============================================================

class ArduinoITO:

    def __init__(
        self,
        port,
        baud=115200,
        timeout=3.0
    ):

        self.port = port

        self.baud = baud

        self.timeout = timeout

        self.serial = None


    # --------------------------------------------------------
    # CONNECT
    # --------------------------------------------------------

    def connect(self):

        import serial

        self.serial = serial.Serial(
            self.port,
            self.baud,
            timeout=self.timeout
        )

        time.sleep(
            2.0
        )

        self._read_available()

        print(
            f"Connected to {self.port}"
        )


    # --------------------------------------------------------
    # READ
    # --------------------------------------------------------

    def _read_available(self):

        while (
            self.serial
            and
            self.serial.in_waiting
        ):

            line = (
                self.serial
                .readline()
                .decode(
                    errors="ignore"
                )
                .strip()
            )

            if line:

                print(
                    "[ARDUINO]",
                    line
                )


    # --------------------------------------------------------
    # SEND
    # --------------------------------------------------------

    def send(
        self,
        line
    ):

        if self.serial is None:

            raise RuntimeError(
                "Arduino is not connected"
            )

        self.serial.write(
            (
                line
                +
                "\n"
            ).encode()
        )


    # --------------------------------------------------------
    # WAIT
    # --------------------------------------------------------

    def wait_for(
        self,
        expected
    ):

        deadline = (
            time.time()
            +
            self.timeout
        )

        while (
            time.time()
            <
            deadline
        ):

            if self.serial.in_waiting:

                response = (
                    self.serial
                    .readline()
                    .decode(
                        errors="ignore"
                    )
                    .strip()
                )

                if response:

                    print(
                        "[ARDUINO]",
                        response
                    )

                    if (
                        response
                        ==
                        expected
                    ):

                        return

                    if response.startswith(
                        "ERROR"
                    ):

                        raise RuntimeError(
                            response
                        )

            time.sleep(
                0.001
            )

        raise TimeoutError(
            "Timed out waiting for "
            + expected
        )


    # --------------------------------------------------------
    # SEND AXIS
    # --------------------------------------------------------

    def send_pattern(
        self,
        pattern
    ):

        pattern = np.asarray(
            pattern
        ).astype(
            np.uint8
        )

        if len(pattern) != AXIS_LENGTH:

            raise ValueError(
                "ITO pattern must contain "
                f"{AXIS_LENGTH} positions"
            )

        self.send(
            f"BEGIN {AXIS_LENGTH}"
        )

        self.wait_for(
            "BEGIN_OK"
        )

        line = "".join(
            "1" if x
            else "0"
            for x in pattern
        )

        self.send(
            line
        )

        self.send(
            "END"
        )

        self.wait_for(
            "ITO_APPLIED"
        )


    # --------------------------------------------------------
    # CLEAR
    # --------------------------------------------------------

    def clear(self):

        self.send_pattern(
            np.zeros(
                AXIS_LENGTH,
                dtype=np.uint8
            )
        )


    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    def close(self):

        if self.serial:

            self.serial.close()

            self.serial = None


# ============================================================
# OPTICAL BENCH
# ============================================================

class OpticalBench:

    def __init__(
        self,
        config,
        seed=None
    ):

        self.cfg = config

        self.source = (
            Correlated32ModeLaser(
                config,
                seed=seed
            )
        )

        self.mode_basis = (
            SpatialModeBasis(
                config,
                seed=seed
            )
        )

        self.ito = (
            ITOAxis(
                config
            )
        )

        self.optics = (
            FocusingOptics(
                config
            )
        )

        self.detector = (
            AreaDetector(
                config
            )
        )

        self.camera = (
            SimulatedCamera(
                config,
                seed=seed
            )
        )


    # --------------------------------------------------------
    # GENERATE OPTICAL FIELD
    # --------------------------------------------------------

    def generate_field(
        self,
        phase_control=0.0
    ):

        self.source.step(
            phase_control
        )

        coefficients = (
            self.source.coefficients()
        )

        field = (
            self.mode_basis.superpose(
                coefficients
            )
        )

        if self.cfg.field_noise > 0:

            noise = (
                self.cfg.field_noise
                *
                (
                    np.random.normal(
                        size=field.shape
                    )
                    +
                    1j
                    *
                    np.random.normal(
                        size=field.shape
                    )
                )
            )

            field += noise

        return field


    # --------------------------------------------------------
    # SINGLE MEASUREMENT
    # --------------------------------------------------------

    def measure(
        self,
        phase_control=0.0
    ):

        # 1. 32 correlated modes

        field = (
            self.generate_field(
                phase_control
            )
        )

        # 2. Generate one-dimensional
        #    ITO control

        pattern = (
            self.ito.pattern_from_field(
                field
            )
        )

        # 3. Expand the 1D ITO into
        #    the optical plane

        modulated = (
            ITOProjection.apply(
                field,
                pattern
            )
        )

        # 4. Focusing optic

        camera_ideal = (
            self.optics.intensity(
                modulated
            )
        )

        # 5. Camera

        camera = (
            self.camera.capture(
                camera_ideal
            )
        )

        # 6. Area-integrated intensity

        area_intensity = (
            self.detector.area_intensity(
                camera
            )
        )

        return {
            "field": field,
            "ito": pattern,
            "camera": camera,
            "area_intensity":
                area_intensity,
            "area_mean":
                self.detector.area_mean(
                    camera
                ),
            "area_variance":
                self.detector.area_variance(
                    camera
                ),
            "area_contrast":
                self.detector.area_contrast(
                    camera
                )
        }


# ============================================================
# SIMULATION
# ============================================================

def run_simulation(
    shots,
    seed
):

    cfg = OpticalConfig()

    bench = OpticalBench(
        cfg,
        seed=seed
    )

    last = None

    print()
    print(
        "=========================================="
    )
    print(
        "32-MODE AREA-INTENSITY SIMULATION"
    )
    print(
        "=========================================="
    )

    print(
        f"Modes       : {cfg.num_modes}"
    )

    print(
        f"ITO axis    : {cfg.axis_length}"
    )

    print(
        f"Camera      : "
        f"{cfg.optical_pixels} x "
        f"{cfg.optical_pixels}"
    )

    print(
        f"ROI         : "
        f"x={cfg.roi_x0}:{cfg.roi_x1}, "
        f"y={cfg.roi_y0}:{cfg.roi_y1}"
    )

    print()

    for n in range(
        shots
    ):

        phase_control = (
            0.02
            *
            np.sin(
                2.0
                *
                np.pi
                *
                n
                /
                25.0
            )
        )

        last = (
            bench.measure(
                phase_control
            )
        )

        if (
            n % 10 == 0
            or
            n == shots - 1
        ):

            print(
                f"{n:5d} | "
                f"I_area="
                f"{last['area_intensity']:.6f} | "
                f"mean="
                f"{last['area_mean']:.6e} | "
                f"variance="
                f"{last['area_variance']:.6e} | "
                f"contrast="
                f"{last['area_contrast']:.6f}"
            )

    plot_simulation(
        last,
        bench.detector
    )


# ============================================================
# PLOT
# ============================================================

def plot_simulation(
    result,
    detector
):

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14, 8)
    )

    # --------------------------------------------------------
    # FIELD AMPLITUDE
    # --------------------------------------------------------

    axes[0, 0].imshow(
        np.abs(
            result["field"]
        )
    )

    axes[0, 0].set_title(
        "32-mode field amplitude"
    )

    # --------------------------------------------------------
    # FIELD PHASE
    # --------------------------------------------------------

    axes[0, 1].imshow(
        np.angle(
            result["field"]
        ),
        cmap="twilight"
    )

    axes[0, 1].set_title(
        "32-mode field phase"
    )

    # --------------------------------------------------------
    # ITO AXIS
    # --------------------------------------------------------

    axes[0, 2].imshow(
        result["ito"][
            None,
            :
        ],
        aspect="auto",
        interpolation="nearest"
    )

    axes[0, 2].set_title(
        "1D ITO axis"
    )

    axes[0, 2].set_xlabel(
        "ITO position"
    )

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    axes[1, 0].imshow(
        result["camera"],
        cmap="gray"
    )

    axes[1, 0].set_title(
        "Camera intensity"
    )

    # ROI

    x0 = detector.cfg.roi_x0
    x1 = detector.cfg.roi_x1

    y0 = detector.cfg.roi_y0
    y1 = detector.cfg.roi_y1

    axes[1, 0].plot(
        [x0, x1, x1, x0, x0],
        [y0, y0, y1, y1, y0],
        linewidth=2
    )

    # --------------------------------------------------------
    # AREA INTENSITY
    # --------------------------------------------------------

    series = (
        detector.series()
    )

    axes[1, 1].plot(
        series
    )

    axes[1, 1].set_title(
        "Area-integrated intensity"
    )

    axes[1, 1].set_xlabel(
        "Measurement"
    )

    axes[1, 1].set_ylabel(
        "I_area"
    )

    # --------------------------------------------------------
    # AUTOCORRELATION
    # --------------------------------------------------------

    if len(series) > 2:

        centered = (
            series
            -
            np.mean(series)
        )

        autocorrelation = (
            np.correlate(
                centered,
                centered,
                mode="full"
            )
        )

        autocorrelation = (
            autocorrelation[
                len(series)-1:
            ]
        )

        if (
            autocorrelation[0]
            >
            EPS
        ):

            autocorrelation /= (
                autocorrelation[0]
            )

        axes[1, 2].plot(
            autocorrelation
        )

    axes[1, 2].set_title(
        "Area-intensity autocorrelation"
    )

    axes[1, 2].set_xlabel(
        "Lag"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# HARDWARE MODE
# ============================================================

def run_hardware(
    port,
    camera_index,
    shots,
    seed
):

    cfg = OpticalConfig()

    bench = OpticalBench(
        cfg,
        seed=seed
    )

    arduino = ArduinoITO(
        port
    )

    camera = RealCamera(
        camera_index
    )

    detector = AreaDetector(
        cfg
    )

    print()
    print(
        "=========================================="
    )
    print(
        "32-MODE HARDWARE AREA-INTENSITY BENCH"
    )
    print(
        "=========================================="
    )

    print(
        f"Arduino     : {port}"
    )

    print(
        f"Camera      : {camera_index}"
    )

    print(
        f"Modes       : {NUM_MODES}"
    )

    print(
        f"ITO axis    : {AXIS_LENGTH}"
    )

    print()

    arduino.connect()

    camera.open()

    try:

        for n in range(
            shots
        ):

            # --------------------------------------------
            # Generate correlated 32-mode field
            # --------------------------------------------

            field = (
                bench.generate_field(
                    phase_control=0.02
                )
            )

            # --------------------------------------------
            # Create 1D ITO pattern
            # --------------------------------------------

            pattern = (
                bench.ito.pattern_from_field(
                    field
                )
            )

            print(
                f"\nSHOT "
                f"{n + 1}/{shots}"
            )

            print(
                "ITO:",
                "".join(
                    str(int(x))
                    for x in pattern
                )
            )

            # --------------------------------------------
            # Send ITO pattern
            # --------------------------------------------

            arduino.send_pattern(
                pattern
            )

            # --------------------------------------------
            # Optical settling
            # --------------------------------------------

            time.sleep(
                0.02
            )

            # --------------------------------------------
            # Camera
            # --------------------------------------------

            frame = (
                camera.capture()
            )

            # --------------------------------------------
            # Area intensity
            # --------------------------------------------

            area_intensity = (
                detector.area_intensity(
                    frame
                )
            )

            area_mean = (
                detector.area_mean(
                    frame
                )
            )

            area_variance = (
                detector.area_variance(
                    frame
                )
            )

            area_contrast = (
                detector.area_contrast(
                    frame
                )
            )

            print(
                f"I_area="
                f"{area_intensity:.4f} | "
                f"mean="
                f"{area_mean:.6e} | "
                f"variance="
                f"{area_variance:.6e} | "
                f"contrast="
                f"{area_contrast:.6f}"
            )

    finally:

        try:

            arduino.clear()

        except Exception as exc:

            print(
                "ITO clear failed:",
                exc
            )

        camera.close()

        arduino.close()

    # --------------------------------------------------------
    # Plot measured area intensity
    # --------------------------------------------------------

    import matplotlib.pyplot as plt

    series = (
        detector.series()
    )

    plt.figure(
        figsize=(10, 4)
    )

    plt.plot(
        series,
        marker="."
    )

    plt.xlabel(
        "Temporal measurement"
    )

    plt.ylabel(
        "Integrated ROI intensity"
    )

    plt.title(
        "Measured entangled/correlated "
        "32-mode intensity over area"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# CORRELATED VS INDEPENDENT
# ============================================================

def compare_correlated_vs_independent(
    shots,
    seed
):

    """
    Compare area-integrated intensity statistics
    between:

        1. strongly correlated modes
        2. independent modes

    The comparison is intentionally performed using
    ONLY the measured area intensity.
    """

    correlated_cfg = (
        OpticalConfig(
            phase_correlation=0.90,
            amplitude_correlation=0.75
        )
    )

    independent_cfg = (
        OpticalConfig(
            phase_correlation=0.0,
            amplitude_correlation=0.0
        )
    )

    correlated = OpticalBench(
        correlated_cfg,
        seed=seed
    )

    independent = OpticalBench(
        independent_cfg,
        seed=seed
    )

    correlated_series = []

    independent_series = []

    for n in range(
        shots
    ):

        a = correlated.measure(
            phase_control=0.02
        )

        b = independent.measure(
            phase_control=0.02
        )

        correlated_series.append(
            a["area_intensity"]
        )

        independent_series.append(
            b["area_intensity"]
        )

    correlated_series = np.asarray(
        correlated_series
    )

    independent_series = np.asarray(
        independent_series
    )

    print()
    print(
        "=========================================="
    )
    print(
        "AREA-INTENSITY STATISTICAL COMPARISON"
    )
    print(
        "=========================================="
    )

    print()

    print(
        "CORRELATED 32-MODE FIELD"
    )

    print(
        "mean     :",
        np.mean(
            correlated_series
        )
    )

    print(
        "variance :",
        np.var(
            correlated_series
        )
    )

    print(
        "std      :",
        np.std(
            correlated_series
        )
    )

    print()

    print(
        "INDEPENDENT 32-MODE FIELD"
    )

    print(
        "mean     :",
        np.mean(
            independent_series
        )
    )

    print(
        "variance :",
        np.var(
            independent_series
        )
    )

    print(
        "std      :",
        np.std(
            independent_series
        )
    )

    print()

    # --------------------------------------------------------
    # Plot distributions
    # --------------------------------------------------------

    import matplotlib.pyplot as plt

    plt.figure(
        figsize=(10, 5)
    )

    plt.hist(
        correlated_series,
        bins=40,
        alpha=0.6,
        label="correlated"
    )

    plt.hist(
        independent_series,
        bins=40,
        alpha=0.6,
        label="independent"
    )

    plt.xlabel(
        "Area-integrated intensity"
    )

    plt.ylabel(
        "Count"
    )

    plt.title(
        "Area-intensity distributions"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "simulate",
            "hardware",
            "compare"
        ],
        default="simulate"
    )

    parser.add_argument(
        "--port",
        default=None
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=100
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234
    )

    args = parser.parse_args()

    if args.mode == "simulate":

        run_simulation(
            shots=args.shots,
            seed=args.seed
        )

    elif args.mode == "hardware":

        if args.port is None:

            raise SystemExit(
                "Hardware mode requires "
                "--port COMx"
            )

        run_hardware(
            port=args.port,
            camera_index=args.camera,
            shots=args.shots,
            seed=args.seed
        )

    elif args.mode == "compare":

        compare_correlated_vs_independent(
            shots=args.shots,
            seed=args.seed
        )


if __name__ == "__main__":

    main()
