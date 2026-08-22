#!/usr/bin/env python3
"""
automatic_ito_optical_qiskit.py

REAL OPTICAL BENCH -> AUTOMATIC ITO PATTERN -> QISKIT

NO POLARISER
NO pattern.txt
NO synthetic optical plane
NO giant 32768 x 32768 NumPy allocation

The physical optical bench performs the optical transformation.

Workflow:

    Python-generated binary pattern
                |
                v
        ITO electrical controller
                |
                v
          physical ITO
                |
                v
        optical propagation
                |
                v
             camera
                |
                v
       measured intensity
                |
                v
        optical mode grid
                |
                v
        optical amplitudes
                |
                v
       compact Qiskit state


Default:

    32 x 32 ITO pattern
    1024 optical modes
    10 logical qubits


IMPORTANT HARDWARE NOTE

The serial packet protocol below is a generic protocol.

Your actual ITO controller may use:

    Arduino
    ESP32
    DAC
    shift registers
    row/column multiplexing
    GPIO
    analogue voltage
    SPI
    I2C

Only send_pattern() needs to be changed to match that
controller.

The optical measurement path remains the same.
"""


from __future__ import annotations

import argparse
import csv
import math
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
# CAMERA
# ============================================================

@dataclass
class CameraConfig:

    index: int = 0

    width: int = 1920

    height: int = 1080

    exposure: Optional[float] = None

    gain: Optional[float] = None

    warmup: int = 20


class OpticalCamera:
    """
    Real camera attached to the optical bench.
    """

    def __init__(
        self,
        config: CameraConfig
    ):

        self.config = config

        self.cap = cv2.VideoCapture(
            config.index
        )

        if not self.cap.isOpened():

            raise RuntimeError(
                f"Cannot open camera {config.index}"
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
            config.warmup
        ):

            self.read()

    # --------------------------------------------------------

    def read(self):

        ok, frame = self.cap.read()

        if not ok or frame is None:

            raise RuntimeError(
                "Camera acquisition failed"
            )

        if frame.ndim == 3:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

        return frame.astype(
            np.float64
        )

    # --------------------------------------------------------

    def close(self):

        if self.cap is not None:

            self.cap.release()

            self.cap = None


# ============================================================
# ITO CONTROLLER
# ============================================================

class ITOController:
    """
    Automatic binary ITO pattern generator/controller.

    No pattern file is required.

    Python creates each pattern in memory and immediately
    transmits it to the physical controller.
    """

    def __init__(
        self,
        port: Optional[str],
        baudrate: int = 115200,
        settle_time: float = 0.05
    ):

        self.port = port

        self.baudrate = baudrate

        self.settle_time = settle_time

        self.serial = None

        if port is not None:

            if not SERIAL_AVAILABLE:

                raise RuntimeError(
                    "Install pyserial:\n"
                    "python -m pip install pyserial"
                )

            self.serial = serial.Serial(
                port,
                baudrate,
                timeout=1
            )

            time.sleep(
                0.5
            )

    # --------------------------------------------------------

    def generate_pattern(
        self,
        rows: int,
        cols: int,
        pattern_type: str,
        index: int = 0
    ) -> np.ndarray:

        """
        Generate a binary optical-electrode pattern.

        Supported:

            single
            checker
            row
            column
            random
            binary
            diagonal
        """

        pattern = np.zeros(
            (
                rows,
                cols
            ),
            dtype=np.uint8
        )

        # ----------------------------------------------------
        # SINGLE ACTIVE PIXEL
        # ----------------------------------------------------

        if pattern_type == "single":

            r = (
                index
                //
                cols
            ) % rows

            c = (
                index
                %
                cols
            )

            pattern[r, c] = 1

        # ----------------------------------------------------
        # CHECKERBOARD
        # ----------------------------------------------------

        elif pattern_type == "checker":

            for r in range(rows):

                for c in range(cols):

                    pattern[r, c] = (
                        (r + c + index)
                        & 1
                    )

        # ----------------------------------------------------
        # ROW SCAN
        # ----------------------------------------------------

        elif pattern_type == "row":

            r = index % rows

            pattern[r, :] = 1

        # ----------------------------------------------------
        # COLUMN SCAN
        # ----------------------------------------------------

        elif pattern_type == "column":

            c = index % cols

            pattern[:, c] = 1

        # ----------------------------------------------------
        # RANDOM
        # ----------------------------------------------------

        elif pattern_type == "random":

            rng = np.random.default_rng(
                index
            )

            pattern[:] = rng.integers(
                0,
                2,
                size=(
                    rows,
                    cols
                ),
                dtype=np.uint8
            )

        # ----------------------------------------------------
        # BINARY INDEX PATTERN
        # ----------------------------------------------------

        elif pattern_type == "binary":

            value = index

            for p in range(
                rows * cols
            ):

                bit = (
                    value
                    >>
                    p
                ) & 1

                r = p // cols

                c = p % cols

                pattern[r, c] = bit

        # ----------------------------------------------------
        # DIAGONAL
        # ----------------------------------------------------

        elif pattern_type == "diagonal":

            offset = index % (
                rows + cols - 1
            )

            for r in range(rows):

                c = offset - r

                if 0 <= c < cols:

                    pattern[r, c] = 1

        else:

            raise ValueError(
                f"Unknown pattern: {pattern_type}"
            )

        return pattern

    # --------------------------------------------------------

    def send_pattern(
        self,
        pattern: np.ndarray
    ):
        """
        Transmit a generated pattern.

        Generic protocol:

            BEGIN rows cols
            010101...
            101010...
            END

        Adapt this method to your actual ITO controller.
        """

        pattern = np.asarray(
            pattern,
            dtype=np.uint8
        )

        rows, cols = pattern.shape

        if self.serial is None:

            print(
                "ITO controller not connected."
            )

            print(
                "Generated pattern:"
            )

            print(
                pattern
            )

            print(
                "Camera will measure the "
                "currently physical optical state."
            )

            time.sleep(
                self.settle_time
            )

            return

        # ----------------------------------------------------
        # GENERIC SERIAL PROTOCOL
        # ----------------------------------------------------

        header = (
            f"BEGIN {rows} {cols}\n"
        )

        self.serial.write(
            header.encode(
                "ascii"
            )
        )

        for row in pattern:

            bits = "".join(
                "1" if x else "0"
                for x in row
            )

            self.serial.write(
                (
                    bits
                    +
                    "\n"
                ).encode(
                    "ascii"
                )
            )

        self.serial.write(
            b"END\n"
        )

        self.serial.flush()

        time.sleep(
            self.settle_time
        )

    # --------------------------------------------------------

    def close(self):

        if self.serial is not None:

            self.serial.close()

            self.serial = None


# ============================================================
# OPTICAL BENCH
# ============================================================

class OpticalBench:

    def __init__(
        self,
        camera: OpticalCamera,
        rows: int,
        cols: int,
        roi=None
    ):

        self.camera = camera

        self.rows = rows

        self.cols = cols

        self.roi = roi

    # --------------------------------------------------------

    def acquire(
        self
    ):

        frame = self.camera.read()

        if self.roi is not None:

            x, y, w, h = self.roi

            frame = frame[
                y:y + h,
                x:x + w
            ]

            if frame.size == 0:

                raise RuntimeError(
                    "Optical ROI is empty"
                )

        return frame

    # --------------------------------------------------------

    def measure_modes(
        self,
        frame
    ):

        """
        Reduce the camera image into a compact optical mode
        grid.

        The large physical optical field is never reconstructed.
        """

        height, width = frame.shape

        modes = np.zeros(
            (
                self.rows,
                self.cols
            ),
            dtype=np.float64
        )

        for r in range(
            self.rows
        ):

            y0 = (
                r
                *
                height
                //
                self.rows
            )

            y1 = (
                (r + 1)
                *
                height
                //
                self.rows
            )

            for c in range(
                self.cols
            ):

                x0 = (
                    c
                    *
                    width
                    //
                    self.cols
                )

                x1 = (
                    (c + 1)
                    *
                    width
                    //
                    self.cols
                )

                region = frame[
                    y0:y1,
                    x0:x1
                ]

                if region.size:

                    modes[
                        r,
                        c
                    ] = region.mean()

        return modes


# ============================================================
# PRIME MASK
# ============================================================

def prime_mask(
    n: int
):

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
        math.isqrt(
            max(
                1,
                n - 1
            )
        ) + 1
    ):

        if mask[p]:

            mask[
                p * p:n:p
            ] = False

    return mask


# ============================================================
# OPTICAL STATE
# ============================================================

def make_optical_state(
    modes,
    prime_only=False
):

    """
    Physical intensity:

        I

    Optical amplitude:

        sqrt(I)

    Quantum probability:

        |psi|^2

    Therefore:

        psi_i = sqrt(I_i / sum(I))
    """

    intensity = np.asarray(
        modes,
        dtype=np.float64
    ).reshape(-1)

    intensity = np.maximum(
        intensity,
        0
    )

    if prime_only:

        intensity *= prime_mask(
            len(intensity)
        )

    total = intensity.sum()

    if total <= 0:

        raise RuntimeError(
            "No optical intensity available."
        )

    probabilities = (
        intensity
        /
        total
    )

    amplitudes = np.sqrt(
        probabilities
    )

    amplitudes /= np.linalg.norm(
        amplitudes
    )

    return (
        amplitudes,
        probabilities
    )


# ============================================================
# QISKIT BACKEND
# ============================================================

class QiskitOpticalBackend:

    def __init__(
        self,
        modes
    ):

        if not QISKIT_AVAILABLE:

            raise RuntimeError(
                "Install Qiskit:\n"
                "python -m pip install -U qiskit"
            )

        if (
            modes
            &
            (modes - 1)
        ):

            raise ValueError(
                "Number of optical modes must "
                "be a power of two."
            )

        self.modes = modes

        self.qubits = (
            modes.bit_length()
            - 1
        )

    # --------------------------------------------------------

    def circuit(
        self,
        amplitudes
    ):

        amplitudes = np.asarray(
            amplitudes,
            dtype=np.complex128
        )

        if len(amplitudes) != self.modes:

            raise ValueError(
                "State size does not match "
                "optical mode count."
            )

        qc = QuantumCircuit(
            self.qubits,
            self.qubits
        )

        # Optical bench -> Qiskit state.
        qc.initialize(
            amplitudes.tolist(),
            range(
                self.qubits
            )
        )

        qc.barrier(
            label="OPTICAL_BENCH"
        )

        qc.measure(
            range(
                self.qubits
            ),
            range(
                self.qubits
            )
        )

        return qc

    # --------------------------------------------------------

    def sample(
        self,
        amplitudes,
        shots,
        seed
    ):

        """
        Sample directly from the compact state.

        This avoids constructing any large physical optical
        matrix.

        For 1024 modes:

            1024 probabilities
            10 logical qubits
        """

        probabilities = (
            np.abs(
                amplitudes
            ) ** 2
        )

        probabilities /= (
            probabilities.sum()
        )

        rng = np.random.default_rng(
            seed
        )

        values = rng.choice(
            self.modes,
            size=shots,
            p=probabilities
        )

        counts = {}

        for value in values:

            value = int(
                value
            )

            bits = format(
                value,
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

def save_run(
    output,
    pattern,
    frame,
    modes,
    amplitudes,
    probabilities,
    counts,
    circuit
):

    out = Path(
        output
    )

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savetxt(
        out / "ito_pattern.csv",
        pattern,
        fmt="%d",
        delimiter=","
    )

    cv2.imwrite(
        str(
            out /
            "optical_camera_frame.png"
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
        "optical_modes.csv",
        modes,
        fmt="%.10g",
        delimiter=","
    )

    np.save(
        out /
        "optical_state.npy",
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

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "bitstring",
                "mode",
                "count"
            ]
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
                    count
                ]
            )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Automatically generate ITO patterns, "
            "measure the real optical bench, "
            "and construct a Qiskit state."
        )
    )

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # OPTICAL GRID
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PATTERN GENERATION
    # --------------------------------------------------------

    parser.add_argument(
        "--pattern",
        choices=[
            "single",
            "checker",
            "row",
            "column",
            "random",
            "binary",
            "diagonal"
        ],
        default="single"
    )

    parser.add_argument(
        "--pattern-index",
        type=int,
        default=0
    )

    parser.add_argument(
        "--sweep",
        type=int,
        default=1,
        help=(
            "Number of automatically generated "
            "ITO patterns to measure."
        )
    )

    # --------------------------------------------------------
    # ITO
    # --------------------------------------------------------

    parser.add_argument(
        "--serial",
        default=None,
        help="ITO controller port, e.g. COM3"
    )

    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200
    )

    parser.add_argument(
        "--settle",
        type=float,
        default=0.05
    )

    # --------------------------------------------------------
    # QISKIT
    # --------------------------------------------------------

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
        "--output",
        default="optical_output"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    modes = (
        args.modes_x
        *
        args.modes_y
    )

    if (
        modes
        &
        (modes - 1)
    ):

        raise ValueError(
            f"{args.modes_x} x {args.modes_y} = "
            f"{modes}; total modes must be a power of two."
        )

    # --------------------------------------------------------
    # HARDWARE
    # --------------------------------------------------------

    camera = OpticalCamera(
        CameraConfig(
            index=args.camera,
            width=args.camera_width,
            height=args.camera_height,
            exposure=args.exposure,
            gain=args.gain
        )
    )

    ito = ITOController(
        port=args.serial,
        baudrate=args.baudrate,
        settle_time=args.settle
    )

    bench = OpticalBench(
        camera=camera,
        rows=args.modes_y,
        cols=args.modes_x,
        roi=(
            tuple(args.roi)
            if args.roi
            else None
        )
    )

    qiskit_backend = (
        QiskitOpticalBackend(
            modes
        )
    )

    try:

        accumulated = np.zeros(
            modes,
            dtype=np.float64
        )

        print()
        print("=" * 72)
        print("AUTOMATIC REAL OPTICAL BENCH")
        print("=" * 72)

        print(
            f"ITO plane              : "
            f"{args.modes_y} x {args.modes_x}"
        )

        print(
            f"Optical modes          : "
            f"{modes}"
        )

        print(
            f"Logical Qiskit qubits  : "
            f"{qiskit_backend.qubits}"
        )

        print(
            "Polariser              : NONE"
        )

        print(
            "Synthetic optical data: NONE"
        )

        print()

        # ----------------------------------------------------
        # AUTOMATIC PATTERN / MEASUREMENT LOOP
        # ----------------------------------------------------

        for sweep_index in range(
            args.sweep
        ):

            pattern = ito.generate_pattern(
                rows=args.modes_y,
                cols=args.modes_x,
                pattern_type=args.pattern,
                index=(
                    args.pattern_index
                    +
                    sweep_index
                )
            )

            print(
                f"[{sweep_index + 1}/{args.sweep}] "
                f"Generating ITO pattern..."
            )

            # Physical electrical operation.
            ito.send_pattern(
                pattern
            )

            print(
                "    Waiting for physical "
                "optical response..."
            )

            time.sleep(
                args.settle
            )

            # ------------------------------------------------
            # REAL OPTICAL MEASUREMENT
            # ------------------------------------------------

            frame = bench.acquire()

            measured = (
                bench.measure_modes(
                    frame
                )
            )

            flat = measured.reshape(
                -1
            )

            accumulated += flat

            print(
                f"    Camera measurement: "
                f"{flat.shape[0]} modes"
            )

            print(
                f"    Total optical power: "
                f"{flat.sum():.6g}"
            )

        # ----------------------------------------------------
        # AVERAGE OPTICAL RESPONSE
        # ----------------------------------------------------

        measured_modes = (
            accumulated
            /
            max(
                args.sweep,
                1
            )
        )

        measured_modes = measured_modes.reshape(
            args.modes_y,
            args.modes_x
        )

        # ----------------------------------------------------
        # OPTICAL STATE
        # ----------------------------------------------------

        amplitudes, probabilities = (
            make_optical_state(
                measured_modes,
                prime_only=args.prime_only
            )
        )

        # ----------------------------------------------------
        # QISKIT
        # ----------------------------------------------------

        circuit = (
            qiskit_backend.circuit(
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

        print()
        print("=" * 72)
        print("OPTICAL STATE")
        print("=" * 72)

        print(
            f"Amplitude elements : "
            f"{len(amplitudes)}"
        )

        print(
            f"Logical qubits     : "
            f"{qiskit_backend.qubits}"
        )

        print(
            f"State norm         : "
            f"{np.linalg.norm(amplitudes):.12f}"
        )

        print(
            f"Probability sum    : "
            f"{probabilities.sum():.12f}"
        )

        print()

        print(
            "Top optical modes:"
        )

        top = np.argsort(
            probabilities
        )[::-1][:32]

        for mode in top:

            bits = format(
                int(mode),
                f"0{qiskit_backend.qubits}b"
            )

            print(
                f"mode={mode:5d} "
                f"|{bits}> "
                f"p={probabilities[mode]:.8f}"
            )

        # ----------------------------------------------------
        # QISKIT RESULTS
        # ----------------------------------------------------

        print()
        print("=" * 72)
        print("QISKIT RESULTS")
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

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        save_run(
            args.output,
            pattern,
            frame,
            measured_modes,
            amplitudes,
            probabilities,
            counts,
            circuit
        )

        print()
        print(
            "=" * 72
        )

        print(
            f"Results saved to: {args.output}"
        )

    finally:

        ito.close()

        camera.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
