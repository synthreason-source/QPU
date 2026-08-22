#!/usr/bin/env python3
"""
automatic_ito_optical_qiskit_temporal.py

REAL OPTICAL BENCH
        +
ITO ELECTRODE PLANE
        +
SPATIAL MODE BINNING
        +
STREAMED TEMPORAL MULTIPLEXING
        +
QISKIT

NO POLARISER
NO pattern.txt
NO giant spatial x temporal NumPy allocation

===============================================================

IMPORTANT MEMORY MODEL
===============================================================

If:

    spatial modes = 32 x 32 = 1024
    temporal bins = 64

the logical optical address space is:

    1024 x 64 = 65,536 modes

BUT THIS PROGRAM DOES NOT CREATE:

    np.zeros(65536)

or:

    np.zeros(1024 * 64)

for the optical field.

Instead:

    temporal bin 0
        measure 1024 spatial modes
        process
        discard

    temporal bin 1
        measure 1024 spatial modes
        process
        discard

    ...

Only one spatial optical plane is resident at a time.

The temporal coordinate is represented as an integer address.

===============================================================

EXAMPLE
===============================================================

32 x 32 spatial plane:

    1024 spatial modes
    10 spatial address bits

with:

    --modes_z 64

gives:

    1024 x 64 = 65,536 optical addresses
    16 address bits

without allocating a 65,536-element optical field.

For:

    --modes_z 1024

you get:

    1,048,576 optical addresses
    20 address bits

again processed temporally.

===============================================================

PRIME MODES
===============================================================

The combined address is:

    mode = temporal_bin * spatial_modes + spatial_mode

Thus the optical bench can report e.g.

    temporal=3
    spatial=1
    mode=3073

and:

    binary = 0000110000000001

If that combined decimal address is prime, it is a
prime-number optical mode.

===============================================================
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


# ==============================================================
# QISKIT
# ==============================================================

try:
    from qiskit import QuantumCircuit

    QISKIT_AVAILABLE = True

except ImportError:
    QISKIT_AVAILABLE = False


# ==============================================================
# OPTIONAL SERIAL
# ==============================================================

try:
    import serial

    SERIAL_AVAILABLE = True

except ImportError:
    SERIAL_AVAILABLE = False


# ==============================================================
# SYNTHETIC CAMERA
# ==============================================================

class SyntheticCamera:
    """
    Synthetic camera.

    This is only for testing.

    It generates random spatial information without generating
    temporal x spatial statevectors.
    """

    def __init__(
        self,
        width=1920,
        height=1080,
        noise=0.03,
        seed=2026,
        structured=True,
    ):
        self.width = int(width)
        self.height = int(height)
        self.noise = float(noise)

        self.rng = np.random.default_rng(seed)

        self.structured = bool(structured)

        self.frame_index = 0

    def read(self):

        h = self.height
        w = self.width

        field = self.rng.random(
            (h, w),
            dtype=np.float64,
        )

        if self.structured:

            yy, xx = np.mgrid[
                0:h,
                0:w,
            ]

            field *= 0.15

            for _ in range(32):

                cx = self.rng.uniform(0, w)
                cy = self.rng.uniform(0, h)

                sx = self.rng.uniform(
                    w * 0.005,
                    w * 0.08,
                )

                sy = self.rng.uniform(
                    h * 0.005,
                    h * 0.08,
                )

                amplitude = self.rng.uniform(
                    0.2,
                    1.0,
                )

                gaussian = np.exp(
                    -(
                        (xx - cx) ** 2
                        /
                        (2 * sx * sx)
                        +
                        (yy - cy) ** 2
                        /
                        (2 * sy * sy)
                    )
                )

                field += amplitude * gaussian

        modulation = (
            0.85
            +
            0.15
            *
            np.sin(
                self.frame_index * 0.37
            )
        )

        field *= modulation

        if self.noise > 0:

            field += self.rng.normal(
                0.0,
                self.noise,
                field.shape,
            )

        field = np.clip(
            field,
            0.0,
            None,
        )

        maximum = field.max()

        if maximum > 0:
            field /= maximum

        frame = (
            field * 255.0
        ).astype(np.uint8)

        self.frame_index += 1

        return frame

    def close(self):
        pass


# ==============================================================
# REAL CAMERA
# ==============================================================

@dataclass
class CameraConfig:

    index: int = 0

    width: int = 1920

    height: int = 1080

    exposure: Optional[float] = None

    gain: Optional[float] = None

    warmup: int = 20


class OpticalCamera:

    def __init__(
        self,
        config: CameraConfig,
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
            config.width,
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            config.height,
        )

        if config.exposure is not None:

            self.cap.set(
                cv2.CAP_PROP_EXPOSURE,
                config.exposure,
            )

        if config.gain is not None:

            self.cap.set(
                cv2.CAP_PROP_GAIN,
                config.gain,
            )

        for _ in range(config.warmup):
            self.read()

    def read(self):

        ok, frame = self.cap.read()

        if not ok or frame is None:

            raise RuntimeError(
                "Camera acquisition failed"
            )

        if frame.ndim == 3:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

        return frame.astype(
            np.float64
        )

    def close(self):

        if self.cap is not None:

            self.cap.release()

            self.cap = None


# ==============================================================
# ITO CONTROLLER
# ==============================================================

class ITOController:

    def __init__(
        self,
        port: Optional[str],
        baudrate=115200,
        settle_time=0.05,
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
                timeout=1,
            )

            time.sleep(0.5)

    # ----------------------------------------------------------
    # PATTERN GENERATOR
    # ----------------------------------------------------------

    def generate_pattern(
        self,
        rows,
        cols,
        pattern_type,
        index=0,
    ):

        pattern = np.zeros(
            (rows, cols),
            dtype=np.uint8,
        )

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

        elif pattern_type == "checker":

            for r in range(rows):

                for c in range(cols):

                    pattern[r, c] = (
                        (r + c + index)
                        &
                        1
                    )

        elif pattern_type == "row":

            r = index % rows

            pattern[r, :] = 1

        elif pattern_type == "column":

            c = index % cols

            pattern[:, c] = 1

        elif pattern_type == "random":

            rng = np.random.default_rng(index)

            pattern[:] = rng.integers(
                0,
                2,
                size=(rows, cols),
                dtype=np.uint8,
            )

        elif pattern_type == "binary":

            value = int(index)

            for p in range(rows * cols):

                pattern[
                    p // cols,
                    p % cols,
                ] = (
                    value >> p
                ) & 1

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

    # ----------------------------------------------------------
    # TRANSMIT
    # ----------------------------------------------------------

    def send_pattern(
        self,
        pattern,
    ):

        pattern = np.asarray(
            pattern,
            dtype=np.uint8,
        )

        rows, cols = pattern.shape

        if self.serial is None:

            print(
                "ITO: no serial controller attached."
            )

            time.sleep(
                self.settle_time
            )

            #return

        self.serial.write(
            f"BEGIN {rows} {cols}\n".encode()
        )

        for row in pattern:

            bits = "".join(
                "1" if x else "0"
                for x in row
            )

            self.serial.write(
                (bits + "\n").encode()
            )

        self.serial.write(
            b"END\n"
        )

        self.serial.flush()

        time.sleep(
            self.settle_time
        )

    def close(self):

        if self.serial is not None:

            self.serial.close()

            self.serial = None


# ==============================================================
# OPTICAL BENCH
# ==============================================================

class OpticalBench:

    """
    The optical bench is the dimensionality-reduction stage.

    It converts:

        camera pixels

    into:

        rows x columns spatial optical modes.

    Crucially, temporal bins are NOT stored here.
    """

    def __init__(
        self,
        camera,
        rows,
        cols,
        roi=None,
    ):

        self.camera = camera

        self.rows = int(rows)

        self.cols = int(cols)

        self.roi = roi

    def acquire(self):

        frame = self.camera.read()

        if self.roi is not None:

            x, y, w, h = self.roi

            frame = frame[
                y:y+h,
                x:x+w,
            ]

            if frame.size == 0:

                raise RuntimeError(
                    "Optical ROI is empty"
                )

        return frame

    def measure_modes(
        self,
        frame,
    ):

        """
        Bin camera pixels into the current spatial optical plane.

        Only rows x cols values are created.
        """

        height, width = frame.shape

        modes = np.zeros(
            (self.rows, self.cols),
            dtype=np.float64,
        )

        for r in range(self.rows):

            y0 = (
                r * height
                //
                self.rows
            )

            y1 = (
                (r + 1) * height
                //
                self.rows
            )

            for c in range(self.cols):

                x0 = (
                    c * width
                    //
                    self.cols
                )

                x1 = (
                    (c + 1) * width
                    //
                    self.cols
                )

                region = frame[
                    y0:y1,
                    x0:x1,
                ]

                if region.size:

                    modes[r, c] = (
                        region.mean()
                    )

        return modes


# ==============================================================
# PRIME TEST
# ==============================================================

def is_prime(n: int) -> bool:

    if n < 2:
        return False

    if n == 2:
        return True

    if n % 2 == 0:
        return False

    d = 3

    while d * d <= n:

        if n % d == 0:
            return False

        d += 2

    return True


# ==============================================================
# PRIME TEST FOR LARGE MODE ADDRESS
# ==============================================================

def prime_mask_small(
    start: int,
    count: int,
):
    """
    Streaming segmented prime sieve.

    This never constructs a mask for the entire temporal space.
    """

    end = start + count

    result = np.ones(
        count,
        dtype=bool,
    )

    if start <= 0 < end:
        result[-start] = False

    if start <= 1 < end:
        result[1 - start] = False

    limit = math.isqrt(
        max(end - 1, 1)
    )

    base = np.ones(
        limit + 1,
        dtype=bool,
    )

    if limit >= 0:
        base[0] = False

    if limit >= 1:
        base[1] = False

    p = 2

    while p * p <= limit:

        if base[p]:

            base[
                p*p:
                limit+1:
                p
            ] = False

        p += 1

    for p in np.flatnonzero(base):

        p = int(p)

        first = max(
            p * p,
            (
                (start + p - 1)
                //
                p
            ) * p,
        )

        if first < end:

            result[
                first - start:
                end - start:
                p
            ] = False

    return result


# ==============================================================
# TEMPORAL ADDRESS
# ==============================================================

def optical_address(
    temporal_bin,
    spatial_mode,
    spatial_modes,
):
    """
    Flatten temporal + spatial coordinates into one logical
    optical address.

        address =
            temporal * spatial_modes
            +
            spatial
    """

    return (
        int(temporal_bin)
        *
        int(spatial_modes)
        +
        int(spatial_mode)
    )


# ==============================================================
# STREAMED TEMPORAL OPTICAL ENGINE
# ==============================================================

class TemporalOpticalEngine:

    """
    Core memory-saving engine.

    It never creates:

        temporal_bins x spatial_modes

    arrays.

    Instead:

        acquire one frame
        measure one spatial plane
        process it
        discard it

    The temporal coordinate exists only as an integer.
    """

    def __init__(
        self,
        bench,
        ito,
        spatial_rows,
        spatial_cols,
        temporal_bins,
        pattern_type,
        pattern_index,
        prime_only=False,
    ):

        self.bench = bench

        self.ito = ito

        self.rows = int(
            spatial_rows
        )

        self.cols = int(
            spatial_cols
        )

        self.temporal_bins = int(
            temporal_bins
        )

        self.pattern_type = (
            pattern_type
        )

        self.pattern_index = int(
            pattern_index
        )

        self.prime_only = bool(
            prime_only
        )

        self.spatial_modes = (
            self.rows * self.cols
        )

        self.total_modes = (
            self.spatial_modes
            *
            self.temporal_bins
        )

        self.qubits = (
            max(
                1,
                (
                    self.total_modes - 1
                ).bit_length()
            )
        )

    def process(
        self,
        save_top=32,
    ):

        """
        Stream the temporal optical field.

        Memory complexity:

            O(spatial_modes)

        rather than:

            O(spatial_modes * temporal_bins)
        """

        top = []

        total_power = 0.0

        prime_power = 0.0

        for t in range(
            self.temporal_bins
        ):

            pattern = (
                self.ito.generate_pattern(
                    rows=self.rows,
                    cols=self.cols,
                    pattern_type=self.pattern_type,
                    index=(
                        self.pattern_index
                        +
                        t
                    ),
                )
            )

            self.ito.send_pattern(
                pattern
            )

            frame = self.bench.acquire()

            spatial = (
                self.bench.measure_modes(
                    frame
                )
            )

            # --------------------------------------------------
            # THIS IS THE ONLY OPTICAL MODE ARRAY.
            # --------------------------------------------------

            flat = spatial.reshape(-1)

            total_power += float(
                flat.sum()
            )

            # --------------------------------------------------
            # Convert only current plane to candidates.
            # --------------------------------------------------

            if self.prime_only:

                # Only the current temporal segment is sieved.
                base = (
                    t
                    *
                    self.spatial_modes
                )

                prime_flags = prime_mask_small(
                    base,
                    self.spatial_modes,
                )

            else:

                prime_flags = None

            for s in range(
                self.spatial_modes
            ):

                intensity = float(
                    flat[s]
                )

                if intensity <= 0:
                    continue

                mode = optical_address(
                    t,
                    s,
                    self.spatial_modes,
                )

                if (
                    prime_flags is not None
                    and
                    not prime_flags[s]
                ):
                    continue

                if is_prime(mode):

                    prime_power += intensity

                    item = (
                        intensity,
                        mode,
                        t,
                        s,
                    )

                    top.append(item)

                    if len(top) > (
                        save_top * 4
                    ):

                        top.sort(
                            reverse=True
                        )

                        del top[
                            save_top:
                        ]

            # --------------------------------------------------
            # CRITICAL:
            #
            # spatial is released before the next temporal bin.
            # --------------------------------------------------

            del spatial
            del flat
            del frame
            del pattern

        top.sort(
            reverse=True
        )

        top = top[
            :save_top
        ]

        return {
            "top_prime_modes": top,
            "total_power": total_power,
            "prime_power": prime_power,
            "spatial_modes": self.spatial_modes,
            "temporal_bins": self.temporal_bins,
            "total_modes": self.total_modes,
            "logical_address_bits": self.qubits,
        }


# ==============================================================
# QISKIT COMPACT REPRESENTATION
# ==============================================================

class QiskitOpticalInterface:

    """
    Qiskit interface.

    IMPORTANT:

    We do not construct a Qiskit statevector with 2^N amplitudes
    for the complete temporal optical address space.

    Instead, each optical address is represented as a computational
    basis address.

    This preserves the optical mode address without asking Qiskit
    to allocate an enormous dense statevector.
    """

    def __init__(
        self,
        total_modes,
    ):

        if not QISKIT_AVAILABLE:

            raise RuntimeError(
                "Install Qiskit:\n"
                "python -m pip install -U qiskit"
            )

        self.total_modes = int(
            total_modes
        )

        self.qubits = max(
            1,
            (
                self.total_modes - 1
            ).bit_length()
        )

    def basis_circuit(
        self,
        mode,
    ):

        mode = int(mode)

        if mode < 0:
            raise ValueError(
                "Mode must be non-negative"
            )

        if mode >= self.total_modes:
            raise ValueError(
                "Mode outside optical space"
            )

        qc = QuantumCircuit(
            self.qubits,
            name="optical_mode",
        )

        bits = format(
            mode,
            f"0{self.qubits}b",
        )

        # Qiskit qubit 0 corresponds to the least significant bit.
        for q, bit in enumerate(
            reversed(bits)
        ):

            if bit == "1":
                qc.x(q)

        qc.barrier(
            label="OPTICAL_BENCH"
        )

        return qc

    def report_mode(
        self,
        mode,
    ):

        mode = int(mode)

        bits = format(
            mode,
            f"0{self.qubits}b",
        )

        return bits


# ==============================================================
# SAVE
# ==============================================================

def save_run(
    output,
    result,
    last_pattern=None,
    last_frame=None,
):

    out = Path(output)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    if last_pattern is not None:

        np.savetxt(
            out / "ito_pattern.csv",
            last_pattern,
            fmt="%d",
            delimiter=",",
        )

    if last_frame is not None:

        cv2.imwrite(
            str(
                out /
                "optical_camera_frame.png"
            ),
            np.clip(
                last_frame,
                0,
                255,
            ).astype(
                np.uint8
            ),
        )

    with (
        out /
        "temporal_prime_modes.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "mode",
                "binary",
                "temporal_bin",
                "spatial_mode",
                "intensity",
            ]
        )

        qubits = result[
            "logical_address_bits"
        ]

        for (
            intensity,
            mode,
            temporal,
            spatial,
        ) in result[
            "top_prime_modes"
        ]:

            writer.writerow(
                [
                    mode,
                    format(
                        mode,
                        f"0{qubits}b",
                    ),
                    temporal,
                    spatial,
                    intensity,
                ]
            )

    metadata = {
        "spatial_modes":
            result["spatial_modes"],
        "temporal_bins":
            result["temporal_bins"],
        "total_modes":
            result["total_modes"],
        "logical_address_bits":
            result["logical_address_bits"],
        "total_power":
            result["total_power"],
        "prime_power":
            result["prime_power"],
    }

    (
        out /
        "run_metadata.txt"
    ).write_text(
        "\n".join(
            f"{k} = {v}"
            for k, v in metadata.items()
        ),
        encoding="utf-8",
    )


# ==============================================================
# REPORT
# ==============================================================

def print_result(
    result,
    top_k,
    spatial_rows,
    spatial_cols,
):

    total_modes = result[
        "total_modes"
    ]

    qubits = result[
        "logical_address_bits"
    ]

    print()
    print("=" * 100)
    print("STREAMED OPTICAL STATE")
    print("=" * 100)

    print(
        f"Spatial plane       : "
        f"{spatial_rows} × {spatial_cols}"
    )

    print(
        f"Spatial modes       : "
        f"{result['spatial_modes']:,}"
    )

    print(
        f"Temporal bins       : "
        f"{result['temporal_bins']:,}"
    )

    print(
        f"Combined optical modes: "
        f"{total_modes:,}"
    )

    print(
        f"Logical address bits: "
        f"{qubits}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The combined temporal/spatial space was NOT "
        "materialized as a NumPy array."
    )

    print(
        "Only one spatial optical plane existed in memory "
        "at each temporal bin."
    )

    print()
    print(
        "PRIME OPTICAL MODES"
    )

    print("-" * 100)

    qiskit_interface = (
        QiskitOpticalInterface(
            total_modes
        )
    )

    for (
        intensity,
        mode,
        temporal,
        spatial,
    ) in result[
        "top_prime_modes"
    ][:top_k]:

        bits = (
            qiskit_interface.report_mode(
                mode
            )
        )

        row = (
            spatial
            //
            spatial_cols
        )

        col = (
            spatial
            %
            spatial_cols
        )

        print(
            f"mode={mode:12d} "
            f"|{bits}> "
            f"prime={mode:<12d} "
            f"temporal={temporal:<8d} "
            f"spatial={spatial:<8d} "
            f"xy=({row:3d},{col:3d}) "
            f"optical={intensity:.8f}"
        )

    print()
    print(
        f"Total optical power : "
        f"{result['total_power']:.8g}"
    )

    print(
        f"Prime-mode power    : "
        f"{result['prime_power']:.8g}"
    )

    if result["total_power"] > 0:

        print(
            f"Prime fraction      : "
            f"{result['prime_power'] / result['total_power']:.8%}"
        )


# ==============================================================
# MAIN
# ==============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Real optical bench with streamed temporal "
            "multiplexing and Qiskit mode addressing."
        )
    )

    # ----------------------------------------------------------
    # CAMERA
    # ----------------------------------------------------------

    parser.add_argument(
        "--synthetic",
        action="store_true",
        help=(
            "Use random synthetic camera "
            "instead of physical camera."
        ),
    )

    parser.add_argument(
        "--synthetic-seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--synthetic-noise",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--camera-width",
        type=int,
        default=1920,
    )

    parser.add_argument(
        "--camera-height",
        type=int,
        default=1080,
    )

    parser.add_argument(
        "--exposure",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--gain",
        type=float,
        default=None,
    )

    # ----------------------------------------------------------
    # SPATIAL OPTICAL PLANE
    # ----------------------------------------------------------

    parser.add_argument(
        "--modes-x",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--modes-y",
        type=int,
        default=32,
    )

    # ----------------------------------------------------------
    # TEMPORAL MULTIPLEXING
    # ----------------------------------------------------------

    parser.add_argument(
        "--modes-z",
        type=int,
        default=1,
        help=(
            "Number of temporal optical bins. "
            "Bins are streamed and never stored simultaneously."
        ),
    )

    # ----------------------------------------------------------
    # ROI
    # ----------------------------------------------------------

    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=(
            "X",
            "Y",
            "WIDTH",
            "HEIGHT",
        ),
    )

    # ----------------------------------------------------------
    # ITO PATTERN
    # ----------------------------------------------------------

    parser.add_argument(
        "--pattern",
        choices=[
            "single",
            "checker",
            "row",
            "column",
            "random",
            "binary",
            "diagonal",
        ],
        default="single",
    )

    parser.add_argument(
        "--pattern-index",
        type=int,
        default=0,
    )

    # ----------------------------------------------------------
    # ITO
    # ----------------------------------------------------------

    parser.add_argument(
        "--serial",
        default=None,
        help="ITO controller COM port.",
    )

    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200,
    )

    parser.add_argument(
        "--settle",
        type=float,
        default=0.05,
    )

    # ----------------------------------------------------------
    # PRIME
    # ----------------------------------------------------------

    parser.add_argument(
        "--prime-only",
        action="store_true",
        help=(
            "Only retain prime-number optical addresses."
        ),
    )

    # ----------------------------------------------------------
    # REPORT
    # ----------------------------------------------------------

    parser.add_argument(
        "--top",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--output",
        default="optical_output",
    )

    args = parser.parse_args()

    # ----------------------------------------------------------
    # VALIDATION
    # ----------------------------------------------------------

    if args.modes_x <= 0:
        raise ValueError(
            "--modes-x must be > 0"
        )

    if args.modes_y <= 0:
        raise ValueError(
            "--modes-y must be > 0"
        )

    if args.modes_z <= 0:
        raise ValueError(
            "--modes_z must be > 0"
        )

    spatial_modes = (
        args.modes_x
        *
        args.modes_y
    )

    total_modes = (
        spatial_modes
        *
        args.modes_z
    )

    logical_bits = max(
        1,
        (
            total_modes - 1
        ).bit_length(),
    )

    # ----------------------------------------------------------
    # CAMERA
    # ----------------------------------------------------------

    if args.synthetic:

        camera = SyntheticCamera(
            width=args.camera_width,
            height=args.camera_height,
            noise=args.synthetic_noise,
            seed=args.synthetic_seed,
            structured=True,
        )

        print(
            "CAMERA MODE        : SYNTHETIC"
        )

    else:

        camera = OpticalCamera(
            CameraConfig(
                index=args.camera,
                width=args.camera_width,
                height=args.camera_height,
                exposure=args.exposure,
                gain=args.gain,
            )
        )

        print(
            "CAMERA MODE        : REAL"
        )

    # ----------------------------------------------------------
    # ITO
    # ----------------------------------------------------------

    ito = ITOController(
        port=args.serial,
        baudrate=args.baudrate,
        settle_time=args.settle,
    )

    # ----------------------------------------------------------
    # OPTICAL BENCH
    # ----------------------------------------------------------

    bench = OpticalBench(
        camera=camera,
        rows=args.modes_y,
        cols=args.modes_x,
        roi=(
            tuple(args.roi)
            if args.roi
            else None
        ),
    )

    try:

        print()
        print("=" * 100)
        print("REAL OPTICAL BENCH — STREAMED TEMPORAL MODE ENGINE")
        print("=" * 100)

        print(
            f"ITO spatial plane   : "
            f"{args.modes_y} × {args.modes_x}"
        )

        print(
            f"Spatial modes       : "
            f"{spatial_modes:,}"
        )

        print(
            f"Temporal modes (--modes_z): "
            f"{args.modes_z:,}"
        )

        print(
            f"Combined addresses  : "
            f"{total_modes:,}"
        )

        print(
            f"Logical address bits: "
            f"{logical_bits}"
        )

        print(
            "Polariser           : NONE"
        )

        print(
            f"Prime-only          : "
            f"{args.prime_only}"
        )

        print()

        print(
            "MEMORY ARCHITECTURE"
        )

        print(
            "    camera frame"
        )

        print(
            "         ↓"
        )

        print(
            "    optical bench"
        )

        print(
            f"    {args.modes_y} × {args.modes_x}"
        )

        print(
            "         ↓"
        )

        print(
            "    process temporal bin"
        )

        print(
            "         ↓"
        )

        print(
            "    discard spatial plane"
        )

        print(
            "         ↓"
        )

        print(
            "    next temporal bin"
        )

        print()

        print(
            "NO temporal statevector is allocated."
        )

        print(
            "NO spatial × temporal NumPy array is allocated."
        )

        print()

        # ------------------------------------------------------
        # STREAM
        # ------------------------------------------------------

        engine = TemporalOpticalEngine(
            bench=bench,
            ito=ito,
            spatial_rows=args.modes_y,
            spatial_cols=args.modes_x,
            temporal_bins=args.modes_z,
            pattern_type=args.pattern,
            pattern_index=args.pattern_index,
            prime_only=args.prime_only,
        )

        result = engine.process(
            save_top=max(
                args.top,
                32,
            ),
        )

        # ------------------------------------------------------
        # REPORT
        # ------------------------------------------------------

        print_result(
            result=result,
            top_k=args.top,
            spatial_rows=args.modes_y,
            spatial_cols=args.modes_x,
        )

        # ------------------------------------------------------
        # QISKIT BASIS REPRESENTATION
        # ------------------------------------------------------

        if QISKIT_AVAILABLE:

            qiskit_interface = (
                QiskitOpticalInterface(
                    total_modes
                )
            )

            print()
            print("=" * 100)
            print("QISKIT OPTICAL MODE REPRESENTATION")
            print("=" * 100)

            print(
                f"Qiskit logical address width: "
                f"{qiskit_interface.qubits} qubits"
            )

            if result[
                "top_prime_modes"
            ]:

                (
                    intensity,
                    mode,
                    temporal,
                    spatial,
                ) = result[
                    "top_prime_modes"
                ][0]

                qc = (
                    qiskit_interface
                    .basis_circuit(
                        mode
                    )
                )

                bits = (
                    qiskit_interface
                    .report_mode(
                        mode
                    )
                )

                print()
                print(
                    f"Prime optical mode: "
                    f"{mode}"
                )

                print(
                    f"Binary address    : "
                    f"|{bits}>"
                )

                print(
                    f"Temporal bin      : "
                    f"{temporal}"
                )

                print(
                    f"Spatial mode      : "
                    f"{spatial}"
                )

                print()
                print(
                    qc.draw(
                        output="text"
                    )
                )

                print()
                print(
                    "This circuit represents the selected optical "
                    "mode as a Qiskit computational-basis address."
                )

        else:

            print()
            print(
                "Qiskit not installed."
            )

            print(
                "Install with:"
            )

            print(
                "python -m pip install -U qiskit"
            )

        # ------------------------------------------------------
        # SAVE
        # ------------------------------------------------------

        save_run(
            output=args.output,
            result=result,
        )

        print()
        print(
            f"Saved results to: {args.output}"
        )

    finally:

        camera.close()

        ito.close()


# ==============================================================
# ENTRY
# ==============================================================

if __name__ == "__main__":
    main()
