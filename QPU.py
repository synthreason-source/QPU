#!/usr/bin/env python3
"""
REAL OPTICAL BENCH -> QISKIT
============================

NO POLARISER.

The physical optical bench is the computational front-end.

Architecture
------------

        LASER / OPTICAL SOURCE
                 |
                 v
       50 mm ITO DEVICE
       electrically driven
                 |
                 v
       FREE-SPACE OPTICAL
          PROPAGATION
                 |
                 v
       CAMERA / DETECTOR
                 |
                 v
       1024 OPTICAL MODES
             32 x 32
                 |
                 v
       OPTICAL STATE VECTOR
             1024
                 |
                 v
          10 QUBIT QISKIT
             STATE
                 |
                 v
            MEASUREMENT


IMPORTANT:

The program does NOT generate a fake optical matrix.

The camera is the source of the optical data.

The large physical optical plane can be arbitrarily large.
Only the measured detector data is retained.


No 32768 x 32768 NumPy matrix is created.

No 32768 x 32768 Qiskit statevector is created.

For 1024 modes:

    1024 = 2^10

so the measured optical state corresponds to 10 logical qubits.

Install:

    python -m pip install numpy opencv-python qiskit

Optional serial ITO control:

    python -m pip install pyserial


Example:

    python optical_bench.py ^
        --camera 0 ^
        --camera-width 1920 ^
        --camera-height 1080 ^
        --modes-x 32 ^
        --modes-y 32 ^
        --shots 8192

With ROI:

    python optical_bench.py ^
        --camera 0 ^
        --roi 100 100 1600 800 ^
        --modes-x 32 ^
        --modes-y 32

Prime optical modes:

    python optical_bench.py ^
        --camera 0 ^
        --prime-only


ITO serial control:

    python optical_bench.py ^
        --camera 0 ^
        --serial COM3 ^
        --pattern pattern.txt

The exact serial protocol of the physical ITO controller is
device-dependent. The serial implementation below sends a
simple ASCII representation and should be adapted to the
actual controller firmware.
"""

from __future__ import annotations

import argparse
import csv
import math
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


# ============================================================
# QISKIT
# ============================================================

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    QISKIT_AVAILABLE = True

except ImportError:

    QISKIT_AVAILABLE = False


# ============================================================
# OPTIONAL SERIAL
# ============================================================

try:

    import serial

    SERIAL_AVAILABLE = True

except ImportError:

    SERIAL_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class CameraConfig:

    camera_index: int = 0

    width: int = 1920
    height: int = 1080

    exposure: Optional[float] = None
    gain: Optional[float] = None

    warmup_frames: int = 20


@dataclass
class OpticalConfig:

    modes_x: int = 32
    modes_y: int = 32

    roi: Optional[tuple[int, int, int, int]] = None

    background_path: Optional[str] = None

    flat_field_path: Optional[str] = None

    prime_only: bool = False

    output_dir: str = "optical_output"


@dataclass
class ITOConfig:

    serial_port: Optional[str] = None

    baudrate: int = 115200

    pattern_path: Optional[str] = None

    settle_time: float = 0.05


# ============================================================
# REAL CAMERA
# ============================================================

class RealOpticalCamera:
    """
    Real camera interface.

    This is the optical detector.

    No synthetic image generation occurs here.
    """

    def __init__(
        self,
        config: CameraConfig
    ):

        self.config = config

        self.cap = cv2.VideoCapture(
            config.camera_index
        )

        if not self.cap.isOpened():

            raise RuntimeError(
                f"Unable to open camera "
                f"{config.camera_index}"
            )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            config.width
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            config.height
        )

        if config.exposure is not None:

            self.cap.set(
                cv2.CAP_PROP_EXPOSURE,
                config.exposure
            )

        if config.gain is not None:

            self.cap.set(
                cv2.CAP_PROP_GAIN,
                config.gain
            )

        for _ in range(
            max(
                0,
                config.warmup_frames
            )
        ):

            self.read()

    # --------------------------------------------------------

    def read(self) -> np.ndarray:

        ok, frame = self.cap.read()

        if not ok or frame is None:

            raise RuntimeError(
                "Optical camera frame acquisition failed"
            )

        # Direct intensity measurement.
        #
        # No polarisation analysis.
        #
        # RGB is reduced to detector intensity.

        if frame.ndim == 3:

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

        else:

            gray = frame

        return gray.astype(
            np.float32,
            copy=False
        )

    # --------------------------------------------------------

    def close(self):

        if self.cap is not None:

            self.cap.release()

            self.cap = None


# ============================================================
# ITO DEVICE
# ============================================================

class ITODevice:
    """
    Interface to the electronically controlled ITO plane.

    The optical calculation occurs physically in the bench.

    This class does NOT emulate the optical effect.

    The serial portion is intentionally simple because the
    exact electrode-controller protocol is specific to the
    user's hardware.
    """

    def __init__(
        self,
        config: ITOConfig
    ):

        self.config = config
        self.connection = None

        if config.serial_port is not None:

            if not SERIAL_AVAILABLE:

                raise RuntimeError(
                    "pyserial is required for ITO control:\n"
                    "python -m pip install pyserial"
                )

            self.connection = serial.Serial(
                config.serial_port,
                config.baudrate,
                timeout=1
            )

            time.sleep(
                config.settle_time
            )

    # --------------------------------------------------------

    def load_pattern(
        self,
        path: str
    ) -> np.ndarray:

        """
        Load a binary electrode pattern.

        Expected format:

            0 1 0 1
            1 0 1 0
            ...

        Returns uint8 binary array.
        """

        values = []

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:

                    continue

                row = []

                for token in (
                    line
                    .replace(",", " ")
                    .split()
                ):

                    value = int(token)

                    if value not in (0, 1):

                        raise ValueError(
                            "ITO pattern must contain "
                            "only 0 or 1"
                        )

                    row.append(value)

                if row:

                    values.append(row)

        if not values:

            raise ValueError(
                "ITO pattern is empty"
            )

        width = len(values[0])

        if any(
            len(row) != width
            for row in values
        ):

            raise ValueError(
                "ITO pattern rows have "
                "different widths"
            )

        return np.asarray(
            values,
            dtype=np.uint8
        )

    # --------------------------------------------------------

    def send_pattern(
        self,
        pattern: np.ndarray
    ):

        """
        Send an electrode pattern to the controller.

        Replace the packet format here with the actual
        controller protocol.
        """

        pattern = np.asarray(
            pattern,
            dtype=np.uint8
        )

        if self.connection is None:

            print(
                "ITO controller not connected; "
                "pattern was loaded but not transmitted."
            )

            return

        rows, cols = pattern.shape

        # Simple framed ASCII protocol:
        #
        # BEGIN rows cols
        # row data
        # ...
        # END
        #
        # This is a generic interface, not a claim about
        # the actual controller's firmware protocol.

        self.connection.write(
            f"BEGIN {rows} {cols}\n".encode()
        )

        for row in pattern:

            line = "".join(
                "1" if x else "0"
                for x in row
            )

            self.connection.write(
                (line + "\n").encode()
            )

        self.connection.write(
            b"END\n"
        )

        self.connection.flush()

        time.sleep(
            self.config.settle_time
        )

    # --------------------------------------------------------

    def close(self):

        if self.connection is not None:

            self.connection.close()

            self.connection = None


# ============================================================
# IMAGE UTILITIES
# ============================================================

def load_gray(
    path: Optional[str]
) -> Optional[np.ndarray]:

    if path is None:

        return None

    image = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise FileNotFoundError(
            path
        )

    return image.astype(
        np.float32
    )


# ============================================================
# OPTICAL BENCH
# ============================================================

class RealOpticalBench:
    """
    Converts the physical camera measurement into a compact
    optical mode vector.

    The bench is NOT simulated.

    The camera supplies the actual optical measurement.
    """

    def __init__(
        self,
        camera: RealOpticalCamera,
        config: OpticalConfig
    ):

        self.camera = camera
        self.config = config

        self.background = load_gray(
            config.background_path
        )

        self.flat_field = load_gray(
            config.flat_field_path
        )

    # --------------------------------------------------------

    @property
    def mode_count(self) -> int:

        return (
            self.config.modes_x
            *
            self.config.modes_y
        )

    # --------------------------------------------------------

    def crop_roi(
        self,
        frame: np.ndarray
    ) -> np.ndarray:

        if self.config.roi is None:

            return frame

        x, y, w, h = (
            self.config.roi
        )

        if (
            x < 0
            or y < 0
            or w <= 0
            or h <= 0
        ):

            raise ValueError(
                "ROI must be X Y WIDTH HEIGHT"
            )

        result = frame[
            y:y + h,
            x:x + w
        ]

        if result.size == 0:

            raise ValueError(
                "ROI is outside camera image"
            )

        return result

    # --------------------------------------------------------

    @staticmethod
    def match(
        image: np.ndarray,
        shape: tuple[int, int]
    ) -> np.ndarray:

        image = np.asarray(
            image,
            dtype=np.float32
        )

        if image.shape == shape:

            return image

        return cv2.resize(
            image,
            (
                shape[1],
                shape[0]
            ),
            interpolation=cv2.INTER_LINEAR
        )

    # --------------------------------------------------------

    def acquire_frame(
        self
    ) -> np.ndarray:

        frame = self.camera.read()

        frame = self.crop_roi(
            frame
        )

        if self.background is not None:

            bg = self.match(
                self.background,
                frame.shape
            )

            frame = frame - bg

        if self.flat_field is not None:

            ff = self.match(
                self.flat_field,
                frame.shape
            )

            frame = frame / np.maximum(
                ff,
                1.0
            )

        return np.maximum(
            frame,
            0.0
        )

    # --------------------------------------------------------

    def decode_modes(
        self,
        frame: np.ndarray
    ) -> np.ndarray:

        """
        Spatially bin the physical camera image.

        The detector image is divided into modes_x x modes_y
        regions.

        Only the compact mode array is retained.
        """

        rows = self.config.modes_y
        cols = self.config.modes_x

        height, width = frame.shape

        modes = np.zeros(
            (
                rows,
                cols
            ),
            dtype=np.float64
        )

        for r in range(rows):

            y0 = (
                r
                *
                height
                //
                rows
            )

            y1 = (
                (r + 1)
                *
                height
                //
                rows
            )

            for c in range(cols):

                x0 = (
                    c
                    *
                    width
                    //
                    cols
                )

                x1 = (
                    (c + 1)
                    *
                    width
                    //
                    cols
                )

                region = frame[
                    y0:y1,
                    x0:x1
                ]

                if region.size:

                    modes[
                        r,
                        c
                    ] = float(
                        region.mean()
                    )

        return modes

    # --------------------------------------------------------

    def acquire_optical_modes(
        self
    ) -> tuple[
        np.ndarray,
        np.ndarray
    ]:

        frame = self.acquire_frame()

        modes = self.decode_modes(
            frame
        )

        return frame, modes


# ============================================================
# PRIME MODES
# ============================================================

def make_prime_mask(
    n: int
) -> np.ndarray:

    mask = np.ones(
        n,
        dtype=bool
    )

    if n > 0:
        mask[0] = False

    if n > 1:
        mask[1] = False

    for p in range(
        2,
        math.isqrt(n - 1) + 1
    ):

        if mask[p]:

            mask[
                p * p:n:p
            ] = False

    return mask


# ============================================================
# OPTICAL INTENSITY -> STATE
# ============================================================

def optical_intensity_to_state(
    modes: np.ndarray,
    prime_only: bool = False
) -> tuple[
    np.ndarray,
    np.ndarray
]:

    """
    Convert measured optical intensities into a real
    normalized quantum state.

    For intensity I:

        amplitude = sqrt(I)

    because:

        probability = |amplitude|^2

    No optical phase is invented.

    The resulting state is therefore the intensity-derived
    real-amplitude state.
    """

    intensity = np.asarray(
        modes,
        dtype=np.float64
    ).reshape(-1)

    intensity = np.maximum(
        intensity,
        0.0
    )

    if prime_only:

        mask = make_prime_mask(
            len(intensity)
        )

        intensity *= mask

    total = float(
        intensity.sum()
    )

    if total <= 0:

        raise RuntimeError(
            "The optical bench produced "
            "zero usable intensity."
        )

    probabilities = (
        intensity
        /
        total
    )

    amplitudes = np.sqrt(
        probabilities
    )

    norm = np.linalg.norm(
        amplitudes
    )

    amplitudes /= norm

    return (
        amplitudes,
        probabilities
    )


# ============================================================
# QISKIT
# ============================================================

class OpticalQiskitBackend:
    """
    Qiskit receives ONLY the compact optical state.

    For 32 x 32 modes:

        1024 amplitudes
        10 qubits

    This is tiny compared with the original physical
    optical plane.
    """

    def __init__(
        self,
        modes: int
    ):

        if not QISKIT_AVAILABLE:

            raise RuntimeError(
                "Install Qiskit with:\n"
                "python -m pip install -U qiskit"
            )

        if modes <= 0:

            raise ValueError(
                "Mode count must be positive."
            )

        if (
            modes
            &
            (modes - 1)
        ) != 0:

            raise ValueError(
                "Mode count must be a power of two "
                "for direct computational-basis mapping."
            )

        self.modes = modes

        self.qubits = (
            modes.bit_length()
            - 1
        )

    # --------------------------------------------------------

    def prepare_state(
        self,
        amplitudes: np.ndarray
    ) -> QuantumCircuit:

        amplitudes = np.asarray(
            amplitudes,
            dtype=np.complex128
        )

        if len(amplitudes) != self.modes:

            raise ValueError(
                "Optical state has incorrect size."
            )

        norm = np.linalg.norm(
            amplitudes
        )

        if norm <= 0:

            raise ValueError(
                "Optical state has zero norm."
            )

        amplitudes = (
            amplitudes
            /
            norm
        )

        circuit = QuantumCircuit(
            self.qubits,
            self.qubits,
            name="REAL_OPTICAL_STATE"
        )

        # Only the compact optical state is transferred.
        #
        # For 1024 modes this is a 1024-element state.
        #
        # It is NOT the 32768 x 32768 physical optical plane.

        circuit.initialize(
            amplitudes.tolist(),
            range(self.qubits)
        )

        circuit.barrier(
            label="OPTICAL_BENCH"
        )

        circuit.measure(
            range(self.qubits),
            range(self.qubits)
        )

        return circuit

    # --------------------------------------------------------

    def sample(
        self,
        amplitudes: np.ndarray,
        shots: int,
        seed: int
    ) -> dict[str, int]:

        """
        Sample the compact optical state.

        Qiskit's Statevector is only 1024 elements for the
        default 32x32 optical mode plane.
        """

        amplitudes = np.asarray(
            amplitudes,
            dtype=np.complex128
        )

        probabilities = (
            np.abs(amplitudes)
            ** 2
        )

        probabilities /= (
            probabilities.sum()
        )

        rng = np.random.default_rng(
            seed
        )

        selected = rng.choice(
            self.modes,
            size=shots,
            p=probabilities
        )

        counts: dict[str, int] = {}

        for mode in selected:

            mode = int(mode)

            bits = format(
                mode,
                f"0{self.qubits}b"
            )

            counts[bits] = (
                counts.get(
                    bits,
                    0
                )
                + 1
            )

        return counts


# ============================================================
# SAVE
# ============================================================

def save_results(
    output_dir: str,
    frame: np.ndarray,
    modes: np.ndarray,
    amplitudes: np.ndarray,
    probabilities: np.ndarray,
    counts: dict[str, int],
    circuit: QuantumCircuit
):

    out = Path(
        output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(
            out /
            "real_optical_camera_frame.png"
        ),
        np.clip(
            frame,
            0,
            255
        ).astype(
            np.uint8
        )
    )

    np.savetxt(
        out /
        "measured_optical_modes.csv",
        modes,
        delimiter=",",
        fmt="%.10g"
    )

    np.save(
        out /
        "optical_state_amplitudes.npy",
        amplitudes
    )

    np.save(
        out /
        "optical_probabilities.npy",
        probabilities
    )

    (
        out /
        "qiskit_circuit.txt"
    ).write_text(
        circuit.draw(
            output="text"
        ),
        encoding="utf-8"
    )

    with (
        out /
        "qiskit_counts.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "bitstring",
                "mode",
                "count",
                "probability"
            ]
        )

        total = max(
            sum(counts.values()),
            1
        )

        for bits, count in sorted(
            counts.items(),
            key=lambda x: x[1],
            reverse=True
        ):

            writer.writerow(
                [
                    bits,
                    int(bits, 2),
                    count,
                    count / total
                ]
            )


# ============================================================
# REPORT
# ============================================================

def report(
    optical_config: OpticalConfig,
    modes: np.ndarray,
    amplitudes: np.ndarray,
    probabilities: np.ndarray,
    counts: dict[str, int],
    qiskit_backend: OpticalQiskitBackend,
    elapsed: float
):

    print()
    print("=" * 72)
    print("REAL OPTICAL BENCH")
    print("=" * 72)

    print(
        "Polariser             : NONE"
    )

    print(
        "Optical source        : REAL"
    )

    print(
        "ITO device            : REAL HARDWARE INPUT"
    )

    print(
        "Camera                : REAL"
    )

    print(
        f"Optical modes         : "
        f"{optical_config.modes_x} x "
        f"{optical_config.modes_y}"
    )

    print(
        f"Total modes           : "
        f"{len(amplitudes)}"
    )

    print(
        f"Logical qubits        : "
        f"{qiskit_backend.qubits}"
    )

    print()
    print(
        "Physical optical plane"
    )

    print(
        "    remains outside the Qiskit state."
    )

    print()
    print(
        "Measured optical matrix:"
    )

    print(
        modes
    )

    print()
    print(
        "Optical state norm:"
    )

    print(
        f"    {np.linalg.norm(amplitudes):.12f}"
    )

    print()
    print(
        "Optical acquisition time:"
    )

    print(
        f"    {elapsed:.4f} seconds"
    )

    print()
    print("=" * 72)
    print("TOP OPTICAL MODES")
    print("=" * 72)

    flat = probabilities.reshape(-1)

    top = np.argsort(
        flat
    )[::-1][:32]

    for mode in top:

        bits = format(
            int(mode),
            f"0{qiskit_backend.qubits}b"
        )

        print(
            f"mode={mode:4d} "
            f"|{bits}> "
            f"p={flat[mode]:.8f}"
        )

    print()
    print("=" * 72)
    print("QISKIT READOUT")
    print("=" * 72)

    total = max(
        sum(counts.values()),
        1
    )

    for bits, count in sorted(
        counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:32]:

        print(
            f"|{bits}> "
            f"count={count:6d} "
            f"p={count / total:.8f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Real optical bench "
            "feeding a compact Qiskit state."
        )
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0
    )

    parser.add_argument(
        "--camera-width",
        type=int,
        default=1920
    )

    parser.add_argument(
        "--camera-height",
        type=int,
        default=1080
    )

    parser.add_argument(
        "--exposure",
        type=float,
        default=None
    )

    parser.add_argument(
        "--gain",
        type=float,
        default=None
    )

    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=(
            "X",
            "Y",
            "WIDTH",
            "HEIGHT"
        )
    )

    parser.add_argument(
        "--modes-x",
        type=int,
        default=32
    )

    parser.add_argument(
        "--modes-y",
        type=int,
        default=32
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=8192
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026
    )

    parser.add_argument(
        "--prime-only",
        action="store_true"
    )

    parser.add_argument(
        "--background",
        default=None
    )

    parser.add_argument(
        "--flat-field",
        default=None
    )

    parser.add_argument(
        "--serial",
        default=None,
        help="ITO controller serial port, e.g. COM3"
    )

    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200
    )

    parser.add_argument(
        "--pattern",
        default=None,
        help="Binary ITO electrode pattern"
    )

    parser.add_argument(
        "--settle",
        type=float,
        default=0.05
    )

    parser.add_argument(
        "--output",
        default="optical_output"
    )

    args = parser.parse_args()

    if not QISKIT_AVAILABLE:

        raise RuntimeError(
            "Qiskit is not installed.\n\n"
            "Install it with:\n"
            "python -m pip install -U qiskit"
        )

    mode_count = (
        args.modes_x
        *
        args.modes_y
    )

    if (
        mode_count
        &
        (mode_count - 1)
    ) != 0:

        raise ValueError(
            "modes_x * modes_y must be a power of two."
        )

    camera_config = CameraConfig(
        camera_index=args.camera,
        width=args.camera_width,
        height=args.camera_height,
        exposure=args.exposure,
        gain=args.gain
    )

    optical_config = OpticalConfig(
        modes_x=args.modes_x,
        modes_y=args.modes_y,
        roi=(
            tuple(args.roi)
            if args.roi
            else None
        ),
        background_path=args.background,
        flat_field_path=args.flat_field,
        prime_only=args.prime_only,
        output_dir=args.output
    )

    ito_config = ITOConfig(
        serial_port=args.serial,
        baudrate=args.baudrate,
        pattern_path=args.pattern,
        settle_time=args.settle
    )

    camera = RealOpticalCamera(
        camera_config
    )

    ito = ITODevice(
        ito_config
    )

    try:

        # ----------------------------------------------------
        # DRIVE REAL ITO DEVICE
        # ----------------------------------------------------

        if args.pattern is not None:

            pattern = ito.load_pattern(
                args.pattern
            )

            print(
                f"Loaded physical ITO pattern: "
                f"{pattern.shape[0]} x "
                f"{pattern.shape[1]}"
            )

            ito.send_pattern(
                pattern
            )

            time.sleep(
                args.settle
            )

        # ----------------------------------------------------
        # REAL OPTICAL ACQUISITION
        # ----------------------------------------------------

        bench = RealOpticalBench(
            camera,
            optical_config
        )

        print()
        print(
            "Acquiring REAL optical field..."
        )

        start = time.perf_counter()

        frame, modes = (
            bench.acquire_optical_modes()
        )

        elapsed = (
            time.perf_counter()
            -
            start
        )

        # ----------------------------------------------------
        # OPTICAL STATE
        # ----------------------------------------------------

        amplitudes, probabilities = (
            optical_intensity_to_state(
                modes,
                prime_only=args.prime_only
            )
        )

        # ----------------------------------------------------
        # QISKIT
        # ----------------------------------------------------

        qiskit_backend = (
            OpticalQiskitBackend(
                mode_count
            )
        )

        circuit = (
            qiskit_backend.prepare_state(
                amplitudes
            )
        )

        counts = (
            qiskit_backend.sample(
                amplitudes,
                shots=args.shots,
                seed=args.seed
            )
        )

        # ----------------------------------------------------
        # REPORT
        # ----------------------------------------------------

        report(
            optical_config,
            modes,
            amplitudes,
            probabilities,
            counts,
            qiskit_backend,
            elapsed
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        save_results(
            args.output,
            frame,
            modes,
            amplitudes,
            probabilities,
            counts,
            circuit
        )

        print()
        print(
            f"Saved results to: {args.output}"
        )

    finally:

        camera.close()
        ito.close()


if __name__ == "__main__":

    main()
