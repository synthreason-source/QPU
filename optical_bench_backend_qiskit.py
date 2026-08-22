#!/usr/bin/env python3

"""
optical_prime_statevector_qiskit.py

OPTICAL BENCH
    |
    +-- 256x256 optical matrix A
    |
    +-- 256x256 optical matrix B
    |
    v
OPTICAL MATRIX MULTIPLICATION
    |
    v
256x256 optical output
    |
    v
65,536 optical modes
    |
    v
PRIME MODE FILTER
    |
    v
65,536-element optical statevector
    |
    v
Qiskit Statevector
    |
    v
16-qubit sampling

IMPORTANT
---------

This version deliberately DOES NOT use:

    QuantumCircuit.initialize(...)

because Qiskit's generic state-preparation circuit can construct
large intermediate matrices.

Instead, the optical bench produces the statevector directly
and Qiskit receives it as a Statevector object.

For 65,536 amplitudes:

    65,536 complex128 values
    = approximately 1 MiB

No 65,536 x 65,536 matrix is created.

Install:

    python -m pip install -U numpy qiskit opencv-python

Run:

    python optical_prime_statevector_qiskit.py --demo --synthetic

"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Any

import argparse
import math
import time

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from qiskit.quantum_info import Statevector

    QISKIT_AVAILABLE = True

except ImportError:
    QISKIT_AVAILABLE = False


# ============================================================
# CONSTANTS
# ============================================================

SIZE = 64

MODE_COUNT = SIZE * SIZE

QUBITS = 16

STATEVECTOR_BYTES = MODE_COUNT * 16


# ============================================================
# CAMERA
# ============================================================

class OpenCVCamera:

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        warmup_frames: int = 12
    ):

        if cv2 is None:

            raise RuntimeError(
                "OpenCV is not installed."
            )

        self.cap = cv2.VideoCapture(
            camera_index
        )

        if not self.cap.isOpened():

            raise RuntimeError(
                f"Unable to open camera "
                f"{camera_index}"
            )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            width
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            height
        )

        for _ in range(
            warmup_frames
        ):

            self.read()

    def read(self):

        ok, frame = self.cap.read()

        if not ok:

            raise RuntimeError(
                "Camera capture failed."
            )

        return cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

    def close(self):

        if self.cap is not None:

            self.cap.release()

            self.cap = None


# ============================================================
# SYNTHETIC OPTICAL PLANE
# ============================================================

class SyntheticOpticalPlane:

    def __init__(
        self,
        size: int = SIZE,
        seed: int = 2026
    ):

        self.size = size

        self.rng = np.random.default_rng(
            seed
        )

    def generate(self):

        size = self.size

        # Optical intensity field.

        plane = self.rng.random(
            (size, size),
            dtype=np.float64
        )

        # Add smooth spatial structure.

        y, x = np.indices(
            (size, size)
        )

        cx = (size - 1) / 2.0
        cy = (size - 1) / 2.0

        r = np.sqrt(
            ((x - cx) / size) ** 2
            +
            ((y - cy) / size) ** 2
        )

        envelope = np.exp(
            -4.0 * r
        )

        plane *= (
            0.25
            +
            envelope
        )

        # Add optical peaks.

        for _ in range(256):

            row = self.rng.integers(
                0,
                size
            )

            col = self.rng.integers(
                0,
                size
            )

            plane[
                row,
                col
            ] += (
                1.0
                +
                10.0
                * self.rng.random()
            )

        return plane


# ============================================================
# OPTICAL BENCH
# ============================================================

class OpticalBench:

    """
    The optical bench is responsible for:

        camera acquisition
        spatial decoding
        matrix multiplication
        prime filtering
        statevector construction

    Qiskit is intentionally absent from this class.
    """

    def __init__(
        self,
        output_shape=(
            SIZE,
            SIZE
        ),
        block_size=32
    ):

        self.output_shape = tuple(
            output_shape
        )

        self.block_size = int(
            block_size
        )

        if self.output_shape != (
            SIZE,
            SIZE
        ):

            raise ValueError(
                "This implementation uses "
                "a 256x256 optical plane."
            )

    # ========================================================
    # GRID DECODING
    # ========================================================

    def decode_frame(
        self,
        frame: np.ndarray
    ):

        frame = np.asarray(
            frame,
            dtype=np.float64
        )

        rows, cols = (
            self.output_shape
        )

        height, width = frame.shape

        matrix = np.zeros(
            (rows, cols),
            dtype=np.float64
        )

        for r in range(rows):

            y0 = round(
                r * height / rows
            )

            y1 = round(
                (r + 1) * height / rows
            )

            for c in range(cols):

                x0 = round(
                    c * width / cols
                )

                x1 = round(
                    (c + 1) * width / cols
                )

                cell = frame[
                    y0:y1,
                    x0:x1
                ]

                if cell.size:

                    matrix[
                        r,
                        c
                    ] = cell.mean()

        return matrix

    # ========================================================
    # OPTICAL MATRIX MULTIPLICATION
    # ========================================================

    def multiply(
        self,
        A: np.ndarray,
        B: np.ndarray
    ):

        """
        Calculate C = A @ B using optical tiles.

        No Qiskit operation occurs here.

        No statevector preparation matrix is created.
        """

        A = np.asarray(
            A,
            dtype=np.float64
        )

        B = np.asarray(
            B,
            dtype=np.float64
        )

        if A.shape != (
            SIZE,
            SIZE
        ):

            raise ValueError(
                "A must be 256x256"
            )

        if B.shape != (
            SIZE,
            SIZE
        ):

            raise ValueError(
                "B must be 256x256"
            )

        C = np.zeros(
            (SIZE, SIZE),
            dtype=np.float64
        )

        bs = self.block_size

        for i in range(
            0,
            SIZE,
            bs
        ):

            i1 = min(
                i + bs,
                SIZE
            )

            for j in range(
                0,
                SIZE,
                bs
            ):

                j1 = min(
                    j + bs,
                    SIZE
                )

                output_block = C[
                    i:i1,
                    j:j1
                ]

                for k in range(
                    0,
                    SIZE,
                    bs
                ):

                    k1 = min(
                        k + bs,
                        SIZE
                    )

                    output_block += (
                        A[
                            i:i1,
                            k:k1
                        ]
                        @
                        B[
                            k:k1,
                            j:j1
                        ]
                    )

        return C

    # ========================================================
    # PRIME SIEVE
    # ========================================================

    @staticmethod
    def primes(
        limit: int
    ):

        mask = np.ones(
            limit + 1,
            dtype=np.bool_
        )

        mask[:2] = False

        root = math.isqrt(
            limit
        )

        for p in range(
            2,
            root + 1
        ):

            if mask[p]:

                mask[
                    p * p:
                    limit + 1:
                    p
                ] = False

        return mask

    # ========================================================
    # PRIME OPTICAL STATEVECTOR
    # ========================================================

    def make_prime_statevector(
        self,
        optical_output: np.ndarray
    ):

        """
        Convert the optical 256x256 plane directly into:

            65,536 complex amplitudes

        Composite modes have amplitude 0.

        Prime modes have amplitude proportional to:

            sqrt(optical intensity)
        """

        optical_output = np.asarray(
            optical_output,
            dtype=np.float64
        )

        if optical_output.shape != (
            SIZE,
            SIZE
        ):

            raise ValueError(
                "Optical output must be "
                "256x256."
            )

        # Optical intensity cannot be negative.

        intensity = np.maximum(
            optical_output,
            0.0
        )

        # Flatten spatial optical plane.

        flat = intensity.reshape(
            MODE_COUNT
        )

        # ----------------------------------------------------
        # PRIME MASK
        # ----------------------------------------------------

        prime = self.primes(
            MODE_COUNT - 1
        )

        # ----------------------------------------------------
        # PRIME-ONLY INTENSITY
        # ----------------------------------------------------

        prime_intensity = np.zeros(
            MODE_COUNT,
            dtype=np.float64
        )

        prime_intensity[
            prime
        ] = flat[
            prime
        ]

        total = float(
            prime_intensity.sum()
        )

        if total <= 0:

            raise RuntimeError(
                "No optical intensity exists "
                "on prime modes."
            )

        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        probability = (
            prime_intensity
            / total
        )

        # ----------------------------------------------------
        # AMPLITUDE
        # ----------------------------------------------------

        state = np.sqrt(
            probability
        ).astype(
            np.complex128
        )

        # Explicit guarantee.

        state[
            ~prime
        ] = 0.0

        # Normalize.

        norm = np.linalg.norm(
            state
        )

        state /= norm

        return state, prime


# ============================================================
# OPTICAL -> QISKIT
# ============================================================

def optical_to_qiskit_state(
    statevector: np.ndarray
):

    """
    CRITICAL:

    Do NOT call:

        QuantumCircuit.initialize()

    because generic state preparation can create a huge
    intermediate unitary.

    Instead Qiskit's Statevector object directly represents
    the 65,536 amplitudes.
    """

    if not QISKIT_AVAILABLE:

        raise RuntimeError(
            "Install Qiskit with:\n"
            "python -m pip install -U qiskit"
        )

    statevector = np.asarray(
        statevector,
        dtype=np.complex128
    )

    if statevector.size != (
        MODE_COUNT
    ):

        raise ValueError(
            "Expected a 65,536-element "
            "statevector."
        )

    norm = np.linalg.norm(
        statevector
    )

    if not np.isclose(
        norm,
        1.0,
        atol=1e-10
    ):

        statevector = (
            statevector
            / norm
        )

    return Statevector(
        statevector
    )


# ============================================================
# QISKIT SAMPLING
# ============================================================

def sample_statevector(
    statevector,
    shots=8192,
    seed=2026
):

    """
    Sample the Statevector directly.

    This avoids constructing a state-preparation circuit.
    """

    probabilities = np.abs(
        statevector.data
    ) ** 2

    rng = np.random.default_rng(
        seed
    )

    samples = rng.choice(
        MODE_COUNT,
        size=shots,
        p=probabilities
    )

    counts = {}

    for value in samples:

        value = int(value)

        bits = format(
            value,
            "016b"
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
# VALIDATE PRIME RESULTS
# ============================================================

def verify_prime_counts(
    counts
):

    invalid = []

    for bits in counts:

        value = int(
            bits,
            2
        )

        if value < 2:

            invalid.append(
                value
            )

            continue

        if value == 2:

            continue

        if value % 2 == 0:

            invalid.append(
                value
            )

            continue

        root = math.isqrt(
            value
        )

        is_prime = True

        for d in range(
            3,
            root + 1,
            2
        ):

            if value % d == 0:

                is_prime = False

                break

        if not is_prime:

            invalid.append(
                value
            )

    return invalid


# ============================================================
# MAIN OPTICAL PROCESS
# ============================================================

def run(
    A,
    B,
    block_size=32,
    shots=8192,
    seed=2026,
    output_dir="optical_prime_output"
):

    print()
    print("=" * 80)
    print(
        "OPTICAL PRIME STATEVECTOR"
    )

    print("=" * 80)
    # --------------------------------------------------------
    # OPTICAL MULTIPLICATION
    # --------------------------------------------------------

    bench = OpticalBench(
        output_shape=(
            SIZE,
            SIZE
        ),
        block_size=block_size
    )

    print()
    print(
        "OPTICAL BENCH:"
    )

    print(
        "Calculating A × B..."
    )

    start = time.perf_counter()

    C = bench.multiply(
        A,
        B
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        "A shape:",
        A.shape
    )

    print(
        "B shape:",
        B.shape
    )

    print(
        "C shape:",
        C.shape
    )

    print(
        "Optical calculation time:",
        f"{elapsed:.6f} seconds"
    )

    # --------------------------------------------------------
    # BUILD STATEVECTOR ON OPTICAL SIDE
    # --------------------------------------------------------

    print()
    print(
        "OPTICAL STATEVECTOR:"
    )

    state, prime = (
        bench.make_prime_statevector(
            C
        )
    )

    prime_count = int(
        np.count_nonzero(
            prime
        )
    )

    print(
        "Statevector elements:",
        len(state)
    )

    print(
        "Prime modes:",
        prime_count
    )

    print(
        "Non-prime modes:",
        MODE_COUNT - prime_count
    )

    print(
        "Norm:",
        np.linalg.norm(state)
    )

    print(
        "Statevector memory:",
        f"{state.nbytes / 1024 / 1024:.3f} MiB"
    )

    # --------------------------------------------------------
    # IMPORTANT MEMORY CHECK
    # --------------------------------------------------------

    dangerous_matrix_bytes = (
        MODE_COUNT
        * MODE_COUNT
        * 16
    )

    print()
    print(
        "MEMORY ARCHITECTURE:"
    )

    print(
        "Actual statevector:",
        f"{state.nbytes / 1024 / 1024:.3f} MiB"
    )

    print(
        "Forbidden state-preparation matrix:",
        f"{dangerous_matrix_bytes / 1024**3:.2f} GiB"
    )

    print(
        "The forbidden matrix is NOT allocated."
    )

    # --------------------------------------------------------
    # PRIME VALIDATION
    # --------------------------------------------------------

    composite_nonzero = np.count_nonzero(
        np.abs(
            state[
                ~prime
            ]
        ) > 1e-15
    )

    print(
        "Non-prime non-zero amplitudes:",
        composite_nonzero
    )

    # --------------------------------------------------------
    # QISKIT STATEVECTOR
    # --------------------------------------------------------

    print()
    print(
        "QISKIT:"
    )

    qstate = (
        optical_to_qiskit_state(
            state
        )
    )

    print(
        "Qiskit dimensions:",
        qstate.dim
    )

    print(
        "Qiskit qubits:",
        qstate.num_qubits
    )

    print(
        "Qiskit state norm:",
        np.linalg.norm(
            qstate.data
        )
    )

    # --------------------------------------------------------
    # SAMPLE
    # --------------------------------------------------------

    print()
    print(
        "Sampling Qiskit Statevector..."
    )

    counts = sample_statevector(
        qstate,
        shots=shots,
        seed=seed
    )

    invalid = (
        verify_prime_counts(
            counts
        )
    )

    print()
    print("=" * 80)
    print(
        "QISKIT PRIME READOUT"
    )
    print("=" * 80)

    print(
        "Shots:",
        shots
    )

    print(
        "Unique states:",
        len(counts)
    )

    print(
        "Non-prime measurements:",
        len(invalid)
    )

    if invalid:

        print(
            "ERROR:",
            invalid[:20]
        )

    else:

        print(
            "Every measured state is prime."
        )

    total = max(
        sum(counts.values()),
        1
    )

    print()

    for bits, count in sorted(
        counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:64]:

        value = int(
            bits,
            2
        )

        print(
            f"|{bits}> "
            f"prime={value:5d} "
            f"count={count:6d} "
            f"p={count / total:.8f}"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    out = Path(
        output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        out / "optical_A.npy",
        A
    )

    np.save(
        out / "optical_B.npy",
        B
    )

    np.save(
        out / "optical_C.npy",
        C
    )

    np.save(
        out / "prime_statevector.npy",
        state
    )

    np.save(
        out / "prime_mask.npy",
        prime
    )

    np.savetxt(
        out / "prime_modes.csv",
        np.flatnonzero(
            prime
        ),
        fmt="%d"
    )

    with (
        out / "qiskit_counts.csv"
    ).open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "bitstring,integer,count,probability\n"
        )

        for bits, count in sorted(
            counts.items(),
            key=lambda x: x[1],
            reverse=True
        ):

            value = int(
                bits,
                2
            )

            f.write(
                f"{bits},"
                f"{value},"
                f"{count},"
                f"{count / total}\n"
            )

    print()
    print(
        "Saved:"
    )

    print(
        out / "optical_C.npy"
    )

    print(
        out / "prime_statevector.npy"
    )

    print(
        out / "prime_mask.npy"
    )

    print(
        out / "qiskit_counts.csv"
    )

    return {
        "optical_output": C,
        "statevector": state,
        "prime_mask": prime,
        "qiskit_state": qstate,
        "counts": counts
    }


# ============================================================
# SYNTHETIC INPUT
# ============================================================

def synthetic_demo(
    seed=2026
):

    rng = np.random.default_rng(
        seed
    )

    A = rng.random(
        (
            SIZE,
            SIZE
        )
    )

    B = rng.random(
        (
            SIZE,
            SIZE
        )
    )

    # Add spatial optical structure.

    for matrix in (
        A,
        B
    ):

        for _ in range(256):

            r = rng.integers(
                0,
                SIZE
            )

            c = rng.integers(
                0,
                SIZE
            )

            matrix[
                r,
                c
            ] += (
                1
                +
                5
                * rng.random()
            )

    return A, B


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "256x256 optical matrix multiplication "
            "to a 65,536-element prime-only "
            "Qiskit Statevector."
        )
    )

    parser.add_argument(
        "--demo",
        action="store_true"
    )

    parser.add_argument(
        "--synthetic",
        action="store_true"
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0
    )

    parser.add_argument(
        "--block-size",
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
        "--output-dir",
        default="optical_prime_output"
    )

    args = parser.parse_args()

    if not args.demo:

        parser.print_help()

        return

    if not QISKIT_AVAILABLE:

        raise RuntimeError(
            "Qiskit is not installed.\n\n"
            "Run:\n"
            "python -m pip install -U qiskit"
        )

    if args.synthetic:

        A, B = synthetic_demo(
            seed=args.seed
        )

        run(
            A=A,
            B=B,
            block_size=args.block_size,
            shots=args.shots,
            seed=args.seed,
            output_dir=args.output_dir
        )

        return

    # --------------------------------------------------------
    # CAMERA MODE
    # --------------------------------------------------------

    camera = OpenCVCamera(
        camera_index=args.camera
    )

    try:

        bench = OpticalBench(
            output_shape=(
                SIZE,
                SIZE
            ),
            block_size=args.block_size
        )

        print(
            "Capture optical plane A..."
        )

        A = bench.decode_frame(
            camera.read()
        )

        print(
            "Capture optical plane B..."
        )

        B = bench.decode_frame(
            camera.read()
        )

        run(
            A=A,
            B=B,
            block_size=args.block_size,
            shots=args.shots,
            seed=args.seed,
            output_dir=args.output_dir
        )

    finally:

        camera.close()


if __name__ == "__main__":

    main()
