#!/usr/bin/env python3
"""
optical_matrix_qiskit_backend.py

OPTICAL BENCH -> LARGE MATRIX MULTIPLICATION -> OPTICAL STATE VECTOR -> QISKIT

Architecture
------------

    Optical plane A
          +
    Optical plane B
          |
          v
    ┌───────────────────────┐
    │   OPTICAL BENCH       │
    │                       │
    │   tiled matrix        │
    │   multiplication      │
    │                       │
    │       C = A @ B       │
    └───────────┬───────────┘
                |
                v
        optical output C
                |
                v
       optical state vector
                |
                v
       strongest-mode
          compression
                |
                v
          Qiskit register
                |
                v
          quantum sampling

The important property is:

    LARGE OPTICAL MATRIX
            !=
    LARGE QISKIT STATEVECTOR

The optical bench can contain thousands/millions of spatial
modes while only the retained optical modes are transferred
to Qiskit.

Examples
--------

Synthetic 64x64:

    python optical_matrix_qiskit_backend.py \
        --demo \
        --synthetic \
        --size 64 \
        --max-modes 256 \
        --shots 8192

Synthetic 256x256:

    python optical_matrix_qiskit_backend.py \
        --demo \
        --synthetic \
        --size 256 \
        --block-size 32 \
        --max-modes 256 \
        --shots 8192

Camera:

    python optical_matrix_qiskit_backend.py \
        --demo \
        --camera 0 \
        --rows 64 \
        --cols 64 \
        --max-modes 256

Requirements:

    python -m pip install -U numpy opencv-python qiskit
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Any

import argparse
import time

import cv2
import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler

    QISKIT_AVAILABLE = True

except ImportError:

    QISKIT_AVAILABLE = False


# ============================================================
# CAMERA CONFIGURATION
# ============================================================

@dataclass
class CameraConfig:

    camera_index: int = 0

    width: int = 1280
    height: int = 720

    exposure: Optional[float] = None
    gain: Optional[float] = None

    warmup_frames: int = 12


# ============================================================
# REAL CAMERA
# ============================================================

class OpenCVCamera:

    def __init__(
        self,
        config: CameraConfig
    ):

        self.cap = cv2.VideoCapture(
            config.camera_index
        )

        if not self.cap.isOpened():

            raise RuntimeError(
                f"Cannot open camera "
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
            max(0, config.warmup_frames)
        ):

            self.read()

    def read(self) -> np.ndarray:

        ok, frame = self.cap.read()

        if not ok or frame is None:

            raise RuntimeError(
                "Camera frame acquisition failed"
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
# SYNTHETIC OPTICAL CAMERA
# ============================================================

class SyntheticCamera:

    """
    Generates a camera image whose spatial cells contain
    the supplied optical matrix.

    This lets the complete optical pipeline be tested without
    physical hardware.
    """

    def __init__(
        self,
        frame_shape=(720, 1280),
        noise_std=1.0,
        seed=2026
    ):

        self.frame_shape = tuple(
            frame_shape
        )

        self.noise_std = float(
            noise_std
        )

        self.rng = np.random.default_rng(
            seed
        )

        self.matrix = None

    def set_matrix(
        self,
        matrix: np.ndarray
    ):

        self.matrix = np.asarray(
            matrix,
            dtype=np.float64
        )

    def read(self) -> np.ndarray:

        if self.matrix is None:

            raise RuntimeError(
                "Synthetic camera matrix "
                "has not been configured"
            )

        rows, cols = self.matrix.shape

        height, width = (
            self.frame_shape
        )

        output = np.full(
            (height, width),
            10.0,
            dtype=np.float64
        )

        maximum = max(
            float(self.matrix.max()),
            1e-12
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

                output[
                    y0:y1,
                    x0:x1
                ] = (
                    10.0
                    +
                    245.0
                    * self.matrix[r, c]
                    / maximum
                )

        if self.noise_std > 0:

            output += self.rng.normal(
                0.0,
                self.noise_std,
                output.shape
            )

        return np.clip(
            output,
            0,
            255
        ).astype(
            np.uint8
        )

    def close(self):

        pass


# ============================================================
# OPTICAL BENCH
# ============================================================

class OpticalBench:

    """
    Camera -> optical matrix decoder.

    The bench is deliberately kept independent of Qiskit.
    """

    def __init__(
        self,
        camera: Any,
        output_shape: Sequence[int],
        roi: Optional[Sequence[int]] = None,
        background: Optional[np.ndarray] = None,
        flat_field: Optional[np.ndarray] = None
    ):

        self.camera = camera

        self.output_shape = tuple(
            int(x)
            for x in output_shape
        )

        if len(self.output_shape) != 2:

            raise ValueError(
                "output_shape must be "
                "(rows, columns)"
            )

        self.roi = (
            tuple(
                int(x)
                for x in roi
            )
            if roi is not None
            else None
        )

        self.background = background
        self.flat_field = flat_field

    # --------------------------------------------------------

    @staticmethod
    def _match(
        image: np.ndarray,
        shape: tuple[int, int]
    ) -> np.ndarray:

        image = np.asarray(
            image,
            dtype=np.float64
        )

        if image.shape == shape:

            return image

        return cv2.resize(
            image,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    # --------------------------------------------------------

    def capture(self) -> np.ndarray:

        frame = self.camera.read()

        frame = np.asarray(
            frame,
            dtype=np.float64
        )

        if self.roi is not None:

            x, y, w, h = self.roi

            if (
                x < 0
                or y < 0
                or w <= 0
                or h <= 0
            ):

                raise ValueError(
                    "ROI must be X Y WIDTH HEIGHT"
                )

            frame = frame[
                y:y+h,
                x:x+w
            ]

            if frame.size == 0:

                raise ValueError(
                    "ROI lies outside camera frame"
                )

        if self.background is not None:

            frame -= self._match(
                self.background,
                frame.shape
            )

        if self.flat_field is not None:

            frame /= np.maximum(
                self._match(
                    self.flat_field,
                    frame.shape
                ),
                1e-12
            )

        return np.maximum(
            frame,
            0.0
        )

    # --------------------------------------------------------

    def decode_grid(
        self,
        frame: np.ndarray
    ) -> np.ndarray:

        rows, cols = self.output_shape

        height, width = frame.shape

        output = np.zeros(
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

                    output[r, c] = (
                        cell.mean()
                    )

        return output

    # --------------------------------------------------------

    def capture_matrix(self):

        frame = self.capture()

        matrix = self.decode_grid(
            frame
        )

        return frame, matrix


# ============================================================
# OPTICAL MATRIX MULTIPLICATION
# ============================================================

class OpticalMatrixMultiplier:

    """
    Large tiled matrix multiplication.

    Conceptually:

        C_ij = sum_k A_ik B_kj

    The operation is performed in optical-backend space
    before Qiskit sees the result.

    Tiling prevents the implementation from unnecessarily
    creating large temporary matrices.
    """

    def __init__(
        self,
        block_size: int = 128
    ):

        if block_size <= 0:

            raise ValueError(
                "block_size must be > 0"
            )

        self.block_size = int(
            block_size
        )

    # --------------------------------------------------------

    def multiply(
        self,
        A: np.ndarray,
        B: np.ndarray
    ) -> np.ndarray:

        A = np.asarray(
            A,
            dtype=np.float64
        )

        B = np.asarray(
            B,
            dtype=np.float64
        )

        if A.ndim != 2:

            raise ValueError(
                "A must be 2-D"
            )

        if B.ndim != 2:

            raise ValueError(
                "B must be 2-D"
            )

        if A.shape[1] != B.shape[0]:

            raise ValueError(
                f"Cannot multiply "
                f"{A.shape} by {B.shape}"
            )

        m, k = A.shape
        _, n = B.shape

        C = np.zeros(
            (m, n),
            dtype=np.float64
        )

        bs = self.block_size

        # ----------------------------------------------------
        # TILED OPTICAL MULTIPLICATION
        # ----------------------------------------------------

        for i in range(
            0,
            m,
            bs
        ):

            i1 = min(
                i + bs,
                m
            )

            for j in range(
                0,
                n,
                bs
            ):

                j1 = min(
                    j + bs,
                    n
                )

                block = C[
                    i:i1,
                    j:j1
                ]

                for p in range(
                    0,
                    k,
                    bs
                ):

                    p1 = min(
                        p + bs,
                        k
                    )

                    block += (
                        A[
                            i:i1,
                            p:p1
                        ]
                        @
                        B[
                            p:p1,
                            j:j1
                        ]
                    )

        return C


# ============================================================
# OPTICAL STATE VECTOR
# ============================================================

class OpticalStateVector:

    """
    Optical output represented as:

        |psi_optical> = sum_i a_i |i>

    with:

        |a_i|^2 = I_i / sum(I)

    The state is stored as a normal NumPy vector.

    Qiskit is not used for this representation.
    """

    def __init__(
        self,
        amplitudes: np.ndarray,
        shape: tuple[int, int]
    ):

        amplitudes = np.asarray(
            amplitudes,
            dtype=np.complex128
        )

        norm = np.linalg.norm(
            amplitudes
        )

        if norm <= 0:

            raise ValueError(
                "Cannot construct optical "
                "state from zero energy"
            )

        self.amplitudes = (
            amplitudes / norm
        )

        self.shape = shape

    # --------------------------------------------------------

    @classmethod
    def from_intensity(
        cls,
        intensity: np.ndarray
    ):

        intensity = np.asarray(
            intensity,
            dtype=np.float64
        )

        intensity = np.maximum(
            intensity,
            0
        )

        total = float(
            intensity.sum()
        )

        if total <= 0:

            raise ValueError(
                "Optical intensity contains "
                "no positive energy"
            )

        probability = (
            intensity.reshape(-1)
            / total
        )

        amplitude = np.sqrt(
            probability
        ).astype(
            np.complex128
        )

        return cls(
            amplitude,
            intensity.shape
        )

    # --------------------------------------------------------

    @property
    def probabilities(self):

        return (
            np.abs(
                self.amplitudes
            ) ** 2
        )

    # --------------------------------------------------------

    @property
    def modes(self):

        return len(
            self.amplitudes
        )

    # --------------------------------------------------------

    def effective_modes(self):

        p = self.probabilities

        return float(
            1.0 / np.sum(p ** 2)
        )

    # --------------------------------------------------------

    def entropy(self):

        p = self.probabilities

        p = p[p > 0]

        return float(
            -np.sum(
                p * np.log2(p)
            )
        )

    # --------------------------------------------------------

    def strongest_modes(
        self,
        number: int
    ):

        number = min(
            int(number),
            self.modes
        )

        return np.argsort(
            self.probabilities
        )[::-1][:number]

    # --------------------------------------------------------

    def compress(
        self,
        max_modes: int
    ):

        indices = (
            self.strongest_modes(
                max_modes
            )
        )

        amplitudes = (
            self.amplitudes[
                indices
            ]
        )

        amplitudes = (
            amplitudes
            / np.linalg.norm(
                amplitudes
            )
        )

        retained_probability = (
            np.sum(
                np.abs(
                    self.amplitudes[
                        indices
                    ]
                ) ** 2
            )
        )

        return OpticalCompressedState(
            amplitudes=amplitudes,
            indices=indices,
            original_modes=self.modes,
            original_shape=self.shape,
            retained_probability=float(
                retained_probability
            )
        )


# ============================================================
# COMPRESSED OPTICAL STATE
# ============================================================

@dataclass
class OpticalCompressedState:

    amplitudes: np.ndarray

    indices: np.ndarray

    original_modes: int

    original_shape: tuple[int, int]

    retained_probability: float

    @property
    def modes(self):

        return len(
            self.amplitudes
        )

    @property
    def required_qubits(self):

        return max(
            1,
            int(
                np.ceil(
                    np.log2(
                        max(
                            self.modes,
                            1
                        )
                    )
                )
            )
        )

    @property
    def qiskit_dimension(self):

        return 2 ** self.required_qubits


# ============================================================
# OPTICAL -> QISKIT
# ============================================================

def optical_state_to_qiskit(
    optical: OpticalCompressedState
):

    if not QISKIT_AVAILABLE:

        raise RuntimeError(
            "Qiskit is not installed.\n"
            "Install with:\n"
            "python -m pip install -U qiskit"
        )

    qubits = (
        optical.required_qubits
    )

    dimension = (
        optical.qiskit_dimension
    )

    amplitudes = np.zeros(
        dimension,
        dtype=np.complex128
    )

    amplitudes[
        :optical.modes
    ] = optical.amplitudes

    amplitudes /= np.linalg.norm(
        amplitudes
    )

    circuit = QuantumCircuit(
        qubits,
        qubits,
        name="optical_state"
    )

    # --------------------------------------------------------
    # The optical state becomes the Qiskit input state.
    # --------------------------------------------------------

    circuit.initialize(
        amplitudes.tolist(),
        range(qubits)
    )

    circuit.barrier(
        label="OPTICAL_STATE"
    )

    circuit.measure(
        range(qubits),
        range(qubits)
    )

    return circuit, {
        "qubits": qubits,
        "dimension": dimension,
        "optical_modes": optical.modes,
        "optical_indices": optical.indices,
        "amplitudes": amplitudes
    }


# ============================================================
# QISKIT SAMPLING
# ============================================================

def run_qiskit_sampler(
    circuit,
    shots: int,
    seed: int = 2026
):

    if not QISKIT_AVAILABLE:

        raise RuntimeError(
            "Qiskit unavailable"
        )

    sampler = StatevectorSampler(
        default_shots=shots,
        seed=seed
    )

    result = sampler.run(
        [circuit],
        shots=shots
    ).result()

    pub = result[0]

    registers = list(
        pub.data
    )

    if not registers:

        raise RuntimeError(
            "No Qiskit classical "
            "register returned"
        )

    register = getattr(
        pub.data,
        registers[0]
    )

    return dict(
        register.get_counts()
    )


# ============================================================
# MEMORY CALCULATIONS
# ============================================================

def complex128_statevector_bytes(
    qubits: int
):

    return (
        (2 ** qubits)
        * 16
    )


def format_bytes(
    value
):

    value = float(value)

    units = [
        "B",
        "KB",
        "MB",
        "GB",
        "TB"
    ]

    for unit in units:

        if value < 1024:

            return f"{value:.3f} {unit}"

        value /= 1024

    return f"{value:.3f} PB"


# ============================================================
# SYNTHETIC MATRICES
# ============================================================

def make_synthetic_matrix(
    rows,
    cols,
    seed
):

    rng = np.random.default_rng(
        seed
    )

    # Positive optical intensity matrix.
    matrix = rng.random(
        (rows, cols)
    )

    # Add several strong optical modes.
    number_of_peaks = max(
        4,
        min(
            32,
            rows * cols // 32
        )
    )

    for _ in range(
        number_of_peaks
    ):

        r = rng.integers(
            0,
            rows
        )

        c = rng.integers(
            0,
            cols
        )

        matrix[r, c] += (
            5.0
            +
            10.0 * rng.random()
        )

    return matrix


# ============================================================
# OPTICAL MATRIX MULTIPLICATION PIPELINE
# ============================================================

def run_optical_pipeline(
    A,
    B,
    block_size,
    max_modes,
    shots
):

    print()
    print("=" * 78)
    print("OPTICAL MATRIX MULTIPLICATION")
    print("=" * 78)

    print(
        "A shape:",
        A.shape
    )

    print(
        "B shape:",
        B.shape
    )

    print(
        "Operation:",
        f"A @ B"
    )

    print(
        "Optical block size:",
        block_size
    )

    # --------------------------------------------------------
    # MATRIX MULTIPLICATION
    # --------------------------------------------------------

    multiplier = (
        OpticalMatrixMultiplier(
            block_size=block_size
        )
    )

    start = time.perf_counter()

    C = multiplier.multiply(
        A,
        B
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        "C shape:",
        C.shape
    )

    print(
        "Optical multiplication time:",
        f"{elapsed:.6f} s"
    )

    # --------------------------------------------------------
    # OPTICAL STATE VECTOR
    # --------------------------------------------------------

    optical_state = (
        OpticalStateVector
        .from_intensity(C)
    )

    print()
    print("=" * 78)
    print("OPTICAL OUTPUT STATE")
    print("=" * 78)

    print(
        "Optical output modes:",
        optical_state.modes
    )

    print(
        "State norm:",
        np.linalg.norm(
            optical_state.amplitudes
        )
    )

    print(
        "Effective optical modes:",
        f"{optical_state.effective_modes():.4f}"
    )

    print(
        "Optical entropy:",
        f"{optical_state.entropy():.6f} bits"
    )

    # --------------------------------------------------------
    # COMPRESS OPTICALLY
    # --------------------------------------------------------

    compressed = (
        optical_state.compress(
            max_modes
        )
    )

    print()
    print("=" * 78)
    print("OPTICAL MODE COMPRESSION")
    print("=" * 78)

    print(
        "Original optical modes:",
        optical_state.modes
    )

    print(
        "Retained modes:",
        compressed.modes
    )

    print(
        "Retained optical probability:",
        f"{compressed.retained_probability:.8f}"
    )

    print(
        "Required Qiskit qubits:",
        compressed.required_qubits
    )

    print(
        "Qiskit state dimension:",
        compressed.qiskit_dimension
    )

    # --------------------------------------------------------
    # MEMORY COMPARISON
    # --------------------------------------------------------

    original_qubits = int(
        np.ceil(
            np.log2(
                optical_state.modes
            )
        )
    )

    original_memory = (
        complex128_statevector_bytes(
            original_qubits
        )
    )

    reduced_memory = (
        complex128_statevector_bytes(
            compressed.required_qubits
        )
    )

    print()
    print("=" * 78)
    print("QISKIT MEMORY")
    print("=" * 78)

    print(
        "If all optical modes entered Qiskit:"
    )

    print(
        "  qubits:",
        original_qubits
    )

    print(
        "  state dimension:",
        2 ** original_qubits
    )

    print(
        "  approximate complex128 memory:",
        format_bytes(
            original_memory
        )
    )

    print()

    print(
        "After optical compression:"
    )

    print(
        "  qubits:",
        compressed.required_qubits
    )

    print(
        "  state dimension:",
        compressed.qiskit_dimension
    )

    print(
        "  approximate complex128 memory:",
        format_bytes(
            reduced_memory
        )
    )

    if reduced_memory > 0:

        print(
            "  memory reduction:",
            f"{original_memory / reduced_memory:.2f}x"
        )

    # --------------------------------------------------------
    # QISKIT
    # --------------------------------------------------------

    circuit, metadata = (
        optical_state_to_qiskit(
            compressed
        )
    )

    print()
    print("=" * 78)
    print("QISKIT OPTICAL STATE")
    print("=" * 78)

    print(
        circuit.draw(
            "text"
        )
    )

    counts = run_qiskit_sampler(
        circuit,
        shots=shots
    )

    print()
    print("=" * 78)
    print("QISKIT READOUT")
    print("=" * 78)

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
            f"count={count:7d} "
            f"p={count / total:.8f}"
        )

    return {
        "A": A,
        "B": B,
        "C": C,
        "optical_state": optical_state,
        "compressed": compressed,
        "circuit": circuit,
        "counts": counts,
        "metadata": metadata
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    output_dir="optical_matrix_output"
):

    output = Path(
        output_dir
    )

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        output / "A.npy",
        results["A"]
    )

    np.save(
        output / "B.npy",
        results["B"]
    )

    np.save(
        output / "C_optical.npy",
        results["C"]
    )

    np.save(
        output / "optical_amplitudes.npy",
        results[
            "optical_state"
        ].amplitudes
    )

    np.save(
        output / "optical_probabilities.npy",
        results[
            "optical_state"
        ].probabilities
    )

    np.save(
        output / "retained_modes.npy",
        results[
            "compressed"
        ].indices
    )

    np.save(
        output / "retained_amplitudes.npy",
        results[
            "compressed"
        ].amplitudes
    )

    np.savetxt(
        output / "qiskit_counts.csv",
        np.array(
            [
                [
                    bits,
                    count
                ]
                for bits, count
                in results[
                    "counts"
                ].items()
            ],
            dtype=object
        ),
        delimiter=",",
        fmt="%s"
    )

    (
        output
        / "qiskit_circuit.txt"
    ).write_text(
        str(
            results[
                "circuit"
            ].draw("text")
        ),
        encoding="utf-8"
    )

    print()
    print(
        "Results saved to:",
        output.resolve()
    )


# ============================================================
# DEMO
# ============================================================

def demo(
    size,
    rows,
    cols,
    block_size,
    max_modes,
    shots,
    seed
):

    if not QISKIT_AVAILABLE:

        raise RuntimeError(
            "Qiskit is required.\n\n"
            "Install with:\n"
            "python -m pip install -U qiskit"
        )

    # --------------------------------------------------------
    # Square synthetic optical planes
    # --------------------------------------------------------

    if size is not None:

        a_rows = size
        a_cols = size

        b_rows = size
        b_cols = size

    else:

        a_rows = rows
        a_cols = cols

        b_rows = cols
        b_cols = rows

    A = make_synthetic_matrix(
        a_rows,
        a_cols,
        seed
    )

    B = make_synthetic_matrix(
        b_rows,
        b_cols,
        seed + 1
    )

    results = run_optical_pipeline(
        A=A,
        B=B,
        block_size=block_size,
        max_modes=max_modes,
        shots=shots
    )

    save_results(
        results
    )


# ============================================================
# CAMERA MODE
# ============================================================

def camera_demo(
    camera_index,
    rows,
    cols,
    block_size,
    max_modes,
    shots,
    roi
):

    if not QISKIT_AVAILABLE:

        raise RuntimeError(
            "Qiskit is required."
        )

    camera = OpenCVCamera(
        CameraConfig(
            camera_index=camera_index
        )
    )

    try:

        bench = OpticalBench(
            camera=camera,
            output_shape=(
                rows,
                cols
            ),
            roi=roi
        )

        print()
        print(
            "Capturing optical matrix A..."
        )

        _, A = bench.capture_matrix()

        print(
            "Capturing optical matrix B..."
        )

        _, B = bench.capture_matrix()

        results = run_optical_pipeline(
            A=A,
            B=B,
            block_size=block_size,
            max_modes=max_modes,
            shots=shots
        )

        save_results(
            results
        )

    finally:

        camera.close()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Large optical matrix multiplication "
            "with Qiskit state preparation."
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
        "--size",
        type=int,
        default=64,
        help=(
            "Square optical matrix size. "
            "Example: 256 means 256x256."
        )
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=64
    )

    parser.add_argument(
        "--cols",
        type=int,
        default=64
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help=(
            "Optical multiplication tile size."
        )
    )

    parser.add_argument(
        "--max-modes",
        type=int,
        default=256,
        help=(
            "Maximum number of optical output "
            "modes transferred to Qiskit."
        )
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
        "--roi",
        type=int,
        nargs=4,
        metavar=(
            "X",
            "Y",
            "W",
            "H"
        )
    )

    args = parser.parse_args()

    if not args.demo:

        parser.print_help()

        return

    if args.synthetic:

        demo(
            size=args.size,
            rows=args.rows,
            cols=args.cols,
            block_size=args.block_size,
            max_modes=args.max_modes,
            shots=args.shots,
            seed=args.seed
        )

    else:

        camera_demo(
            camera_index=args.camera,
            rows=args.rows,
            cols=args.cols,
            block_size=args.block_size,
            max_modes=args.max_modes,
            shots=args.shots,
            roi=args.roi
        )


if __name__ == "__main__":

    main()
