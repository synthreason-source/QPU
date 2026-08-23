#!/usr/env python3
"""
General quantum computer via Qiskit + optical bench.

Backends:
  - simulator: Qiskit AerSimulator
  - optical:   real optical bench (camera + ITO + temporal streaming)

Usage examples:

  python quantum_optical_bench.py \
      --backend simulator \
      --qubits 10 \
      --shots 2048

  python quantum_optical_bench.py \
      --backend optical \
      --synthetic \
      --qubits 10 \
      --modes-x 32 \
      --modes-y 32 \
      --modes-z 4 \
      --shots 1024

  python quantum_optical_bench.py \
      --script circuit.q \
      --qubits 2
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

# ===============================================================
# QISKIT
# ===============================================================

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    QISKIT_AVAILABLE = True

except ImportError:
    QISKIT_AVAILABLE = False


# ===============================================================
# OPTIONAL SERIAL
# ===============================================================

try:
    import serial

    SERIAL_AVAILABLE = True

except ImportError:
    SERIAL_AVAILABLE = False


# ===============================================================
# SYNTHETIC CAMERA
# ===============================================================

class SyntheticCamera:
    """
    Synthetic camera for testing.
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
            yy, xx = np.mgrid[0:h, 0:w]
            field *= 0.15

            for _ in range(32):
                cx = self.rng.uniform(0, w)
                cy = self.rng.uniform(0, h)

                sx = self.rng.uniform(w * 0.005, w * 0.08)
                sy = self.rng.uniform(h * 0.005, h * 0.08)

                amplitude = self.rng.uniform(0.2, 1.0)

                gaussian = np.exp(
                    -(
                        (xx - cx) ** 2 / (2 * sx * sx)
                        + (yy - cy) ** 2 / (2 * sy * sy)
                    )
                )

                field += amplitude * gaussian

        modulation = 0.85 + 0.15 * np.sin(self.frame_index * 0.37)
        field *= modulation

        if self.noise > 0:
            field += self.rng.normal(0.0, self.noise, field.shape)

        field = np.clip(field, 0.0, None)

        maximum = field.max()
        if maximum > 0:
            field /= maximum

        frame = (field * 255.0).astype(np.uint8)
        self.frame_index += 1

        return frame

    def close(self):
        pass


# ===============================================================
# REAL CAMERA
# ===============================================================

@dataclass
class CameraConfig:
    index: int = 0
    width: int = 1920
    height: int = 1080
    exposure: Optional[float] = None
    gain: Optional[float] = None
    warmup: int = 20


class OpticalCamera:
    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap = cv2.VideoCapture(config.index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {config.index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)

        if config.exposure is not None:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, config.exposure)

        if config.gain is not None:
            self.cap.set(cv2.CAP_PROP_GAIN, config.gain)

        for _ in range(config.warmup):
            self.read()

    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Camera acquisition failed")

        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame.astype(np.float64)

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ===============================================================
# ITO CONTROLLER
# ===============================================================

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
                    "Install pyserial:\npython -m pip install pyserial"
                )

            self.serial = serial.Serial(port, baudrate, timeout=1)
            time.sleep(0.5)

    def generate_pattern(
        self,
        rows,
        cols,
        pattern_type,
        index=0,
    ):
        pattern = np.zeros((rows, cols), dtype=np.uint8)

        if pattern_type == "single":
            r = (index // cols) % rows
            c = index % cols
            pattern[r, c] = 1

        elif pattern_type == "checker":
            for r in range(rows):
                for c in range(cols):
                    pattern[r, c] = ((r + c + index) & 1)

        elif pattern_type == "row":
            r = index % rows
            pattern[r, :] = 1

        elif pattern_type == "column":
            c = index % cols
            pattern[:, c] = 1

        elif pattern_type == "random":
            rng = np.random.default_rng(index)
            pattern[:] = rng.integers(0, 2, size=(rows, cols), dtype=np.uint8)

        elif pattern_type == "binary":
            value = int(index)
            for p in range(rows * cols):
                pattern[p // cols, p % cols] = (value >> p) & 1

        elif pattern_type == "diagonal":
            offset = index % (rows + cols - 1)
            for r in range(rows):
                c = offset - r
                if 0 <= c < cols:
                    pattern[r, c] = 1

        else:
            raise ValueError(f"Unknown pattern: {pattern_type}")

        return pattern

    def send_pattern(self, pattern):
        pattern = np.asarray(pattern, dtype=np.uint8)
        rows, cols = pattern.shape

        if self.serial is None:
            print("ITO: no serial controller attached.")
            time.sleep(self.settle_time)
            return

        self.serial.write(f"BEGIN {rows} {cols}\n".encode())

        for row in pattern:
            bits = "".join("1" if x else "0" for x in row)
            self.serial.write((bits + "\n").encode())

        self.serial.write(b"END\n")
        self.serial.flush()
        time.sleep(self.settle_time)

    def close(self):
        if self.serial is not None:
            self.serial.close()
            self.serial = None


# ===============================================================
# OPTICAL BENCH
# ===============================================================

class OpticalBench:
    """
    Converts camera pixels into rows x cols spatial optical modes.
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
            frame = frame[y:y+h, x:x+w]
            if frame.size == 0:
                raise RuntimeError("Optical ROI is empty")

        return frame

    def measure_modes(self, frame):
        height, width = frame.shape
        modes = np.zeros((self.rows, self.cols), dtype=np.float64)

        for r in range(self.rows):
            y0 = (r * height) // self.rows
            y1 = ((r + 1) * height) // self.rows

            for c in range(self.cols):
                x0 = (c * width) // self.cols
                x1 = ((c + 1) * width) // self.cols

                region = frame[y0:y1, x0:x1]
                if region.size:
                    modes[r, c] = region.mean()

        return modes


# ===============================================================
# PRIME TEST (OPTIONAL ANALYSIS)
# ===============================================================

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


def prime_mask_small(start: int, count: int):
    end = start + count
    result = np.ones(count, dtype=bool)

    if start <= 0 < end:
        result[-start] = False
    if start <= 1 < end:
        result[1 - start] = False

    limit = math.isqrt(max(end - 1, 1))
    base = np.ones(limit + 1, dtype=bool)

    if limit >= 0:
        base[0] = False
    if limit >= 1:
        base[1] = False

    p = 2
    while p * p <= limit:
        if base[p]:
            base[p*p:limit+1:p] = False
        p += 1

    for p in np.flatnonzero(base):
        p = int(p)
        first = max(
            p * p,
            ((start + p - 1) // p) * p,
        )
        if first < end:
            result[first - start:end - start:p] = False

    return result


def optical_address(temporal_bin, spatial_mode, spatial_modes):
    return int(temporal_bin) * int(spatial_modes) + int(spatial_mode)


# ===============================================================
# STREAMED TEMPORAL OPTICAL ENGINE
# ===============================================================

class TemporalOpticalEngine:
    """
    Streams temporal optical field without allocating
    temporal_bins x spatial_modes arrays.
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
        self.rows = int(spatial_rows)
        self.cols = int(spatial_cols)
        self.temporal_bins = int(temporal_bins)
        self.pattern_type = pattern_type
        self.pattern_index = int(pattern_index)
        self.prime_only = bool(prime_only)

        self.spatial_modes = self.rows * self.cols
        self.total_modes = self.spatial_modes * self.temporal_bins
        self.qubits = max(1, (self.total_modes - 1).bit_length())

    def process(self, save_top=32):
        top = []
        total_power = 0.0
        prime_power = 0.0

        for t in range(self.temporal_bins):
            pattern = self.ito.generate_pattern(
                rows=self.rows,
                cols=self.cols,
                pattern_type=self.pattern_type,
                index=(self.pattern_index + t),
            )

            self.ito.send_pattern(pattern)
            frame = self.bench.acquire()
            spatial = self.bench.measure_modes(frame)

            flat = spatial.reshape(-1)
            total_power += float(flat.sum())

            if self.prime_only:
                base = t * self.spatial_modes
                prime_flags = prime_mask_small(base, self.spatial_modes)
            else:
                prime_flags = None

            for s in range(self.spatial_modes):
                intensity = float(flat[s])
                if intensity <= 0:
                    continue

                mode = optical_address(t, s, self.spatial_modes)

                if prime_flags is not None and not prime_flags[s]:
                    continue

                if is_prime(mode):
                    prime_power += intensity
                    item = (intensity, mode, t, s)
                    top.append(item)

                    if len(top) > (save_top * 4):
                        top.sort(reverse=True)
                        del top[save_top:]

            del spatial, flat, frame, pattern

        top.sort(reverse=True)
        top = top[:save_top]

        return {
            "top_prime_modes": top,
            "total_power": total_power,
            "prime_power": prime_power,
            "spatial_modes": self.spatial_modes,
            "temporal_bins": self.temporal_bins,
            "total_modes": self.total_modes,
            "logical_address_bits": self.qubits,
        }


# ===============================================================
# QISKIT OPTICAL INTERFACE (BASIS STATES)
# ===============================================================

class QiskitOpticalInterface:
    """
    Represents optical addresses as Qiskit computational-basis states.
    """

    def __init__(self, total_modes, num_qubits=None):
        if not QISKIT_AVAILABLE:
            raise RuntimeError(
                "Install Qiskit:\npython -m pip install -U qiskit"
            )

        self.total_modes = int(total_modes)
        self.qubits = int(num_qubits) if num_qubits is not None else max(1, (self.total_modes - 1).bit_length())

    def basis_circuit(self, mode):
        mode = int(mode)
        if mode < 0:
            raise ValueError("Mode must be non-negative")
        if mode >= self.total_modes:
            raise ValueError("Mode outside optical space")

        qc = QuantumCircuit(self.qubits, name="optical_mode")
        bits = format(mode, f"0{self.qubits}b")[-self.qubits:]

        for q, bit in enumerate(reversed(bits)):
            if bit == "1":
                qc.x(q)

        qc.barrier(label="OPTICAL_BENCH")
        return qc

    def report_mode(self, mode):
        mode = int(mode)
        bits = format(mode, "b")
        if len(bits) > self.qubits:
            return bits[-self.qubits:]
        return bits.zfill(self.qubits)


# ===============================================================
# GENERAL QUANTUM COMPUTER (QISKIT + OPTICAL BACKEND)
# ===============================================================

@dataclass
class OpticalRunResult:
    counts: dict[str, int]
    shots: int
    circuit: QuantumCircuit


class GeneralQuantumComputer:
    """
    Qiskit-compatible control layer.

    The optical bench is used as the physical execution backend.
    AerSimulator is used for development and validation.
    """

    def __init__(
        self,
        num_qubits: int,
        optical_backend=None,
        seed: Optional[int] = 2026,
    ):
        if num_qubits <= 0:
            raise ValueError("num_qubits must be positive")

        self.num_qubits = int(num_qubits)
        self.optical_backend = optical_backend
        self.seed = seed

    def build_circuit(
        self,
        circuit_text: Optional[str] = None,
    ) -> QuantumCircuit:
        qc = QuantumCircuit(
            self.num_qubits,
            self.num_qubits,
        )

        if circuit_text is None:
            qc.h(0)
            for q in range(1, self.num_qubits):
                qc.cx(0, q)
        else:
            namespace = {
                "qc": qc,
                "QuantumCircuit": QuantumCircuit,
            }
            exec(circuit_text, {}, namespace)

        qc.measure(range(self.num_qubits), range(self.num_qubits))
        return qc

    def run_simulator(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
    ) -> OpticalRunResult:
        if not QISKIT_AVAILABLE:
            raise RuntimeError(
                "Qiskit or Qiskit-Aer not available"
            )

        simulator = AerSimulator(seed_simulator=self.seed)
        compiled = transpile(circuit, simulator)
        job = simulator.run(compiled, shots=shots)
        counts = job.result().get_counts()

        return OpticalRunResult(
            counts=counts,
            shots=shots,
            circuit=compiled,
        )

    def run_optical(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
    ) -> OpticalRunResult:
        if self.optical_backend is None:
            raise RuntimeError(
                "No optical backend has been attached"
            )

        counts = self.optical_backend.execute(circuit, shots=shots)

        return OpticalRunResult(
            counts=counts,
            shots=shots,
            circuit=circuit,
        )


def parse_operations(
    operations: list[str],
    num_qubits: int,
) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, num_qubits)

    for operation in operations:
        fields = operation.split(":")
        name = fields[0].lower()

        if name in {"h", "x", "y", "z"}:
            q = int(fields[1])
            getattr(qc, name)(q)

        elif name in {"s", "t"}:
            q = int(fields[1])
            getattr(qc, name)(q)

        elif name in {"cx", "cz", "swap"}:
            q0 = int(fields[1])
            q1 = int(fields[2])
            getattr(qc, name)(q0, q1)

        elif name in {"rx", "ry", "rz"}:
            angle = float(fields[1])
            q = int(fields[2])
            getattr(qc, name)(angle, q)

        elif name == "u":
            theta = float(fields[1])
            phi = float(fields[2])
            lam = float(fields[3])
            q = int(fields[4])
            qc.u(theta, phi, lam, q)

        else:
            raise ValueError(f"Unsupported operation: {operation}")

    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def parse_q_file(file_path: str | Path) -> list[str]:
    """
    Parses a .q script file, returning a list of operation strings.
    Ignores empty lines and comments starting with '#'.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Quantum script file not found: {path}")

    operations = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            operations.append(line)
    return operations


# ===============================================================
# OPTICAL BACKEND WRAPPER
# ===============================================================

class OpticalBackend:
    """
    Hardware abstraction for the optical bench.
    """

    def __init__(
        self,
        bench: OpticalBench,
        ito: ITOController,
        num_qubits: int,
        spatial_rows: int,
        spatial_cols: int,
        temporal_bins: int = 1,
        pattern_type: str = "single",
        prime_only: bool = False,
    ):
        self.bench = bench
        self.ito = ito
        self.num_qubits = num_qubits
        self.spatial_rows = spatial_rows
        self.spatial_cols = spatial_cols
        self.temporal_bins = temporal_bins
        self.pattern_type = pattern_type
        self.prime_only = prime_only

        self.spatial_modes = spatial_rows * spatial_cols
        self.total_modes = self.spatial_modes * self.temporal_bins
        self.address_bits = max(1, (self.total_modes - 1).bit_length())

    def execute(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
    ) -> dict[str, int]:
        engine = TemporalOpticalEngine(
            bench=self.bench,
            ito=self.ito,
            spatial_rows=self.spatial_rows,
            spatial_cols=self.spatial_cols,
            temporal_bins=self.temporal_bins,
            pattern_type=self.pattern_type,
            pattern_index=0,
            prime_only=self.prime_only,
        )

        result = engine.process(save_top=2**self.num_qubits)

        counts = {}
        qiskit_interface = QiskitOpticalInterface(self.total_modes, num_qubits=self.num_qubits)

        for intensity, mode, temporal, spatial in result["top_prime_modes"]:
            bits = qiskit_interface.report_mode(mode)
            counts[bits] = counts.get(bits, 0) + int(round(intensity))

        total = sum(counts.values())
        if total == 0:
            return {"0" * self.num_qubits: shots}

        scaled = {}
        for k, v in counts.items():
            scaled[k] = max(1, int(round(v * shots / total)))

        return scaled


# ===============================================================
# SAVE & REPORT
# ===============================================================

def save_run(
    output,
    result,
    last_pattern=None,
    last_frame=None,
):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    if last_pattern is not None:
        np.savetxt(
            out / "ito_pattern.csv",
            last_pattern,
            fmt="%d",
            delimiter=",",
        )

    if last_frame is not None:
        cv2.imwrite(
            str(out / "optical_camera_frame.png"),
            np.clip(last_frame, 0, 255).astype(np.uint8),
        )

    with (out / "temporal_prime_modes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "mode", "binary", "temporal_bin", "spatial_mode", "intensity"
        ])

        qubits = result["logical_address_bits"]

        for intensity, mode, temporal, spatial in result["top_prime_modes"]:
            writer.writerow([
                mode,
                format(mode, f"0{qubits}b"),
                temporal,
                spatial,
                intensity,
            ])

    metadata = {
        "spatial_modes": result["spatial_modes"],
        "temporal_bins": result["temporal_bins"],
        "total_modes": result["total_modes"],
        "logical_address_bits": result["logical_address_bits"],
        "total_power": result["total_power"],
        "prime_power": result["prime_power"],
    }

    (out / "run_metadata.txt").write_text(
        "\n".join(f"{k} = {v}" for k, v in metadata.items()),
        encoding="utf-8",
    )


def print_result(result, top_k, spatial_rows, spatial_cols, num_qubits=None):
    total_modes = result["total_modes"]
    qubits = num_qubits if num_qubits is not None else result["logical_address_bits"]

    print()
    print("=" * 100)
    print("STREAMED OPTICAL STATE")
    print("=" * 100)

    print(f"Spatial plane         : {spatial_rows} × {spatial_cols}")
    print(f"Spatial modes         : {result['spatial_modes']:,}")
    print(f"Temporal bins         : {result['temporal_bins']:,}")
    print(f"Combined optical modes: {total_modes:,}")
    print(f"Logical address bits  : {qubits}")

    print()
    print("IMPORTANT:")
    print("The combined temporal/spatial space was NOT materialized as a NumPy array.")
    print("Only one spatial optical plane existed in memory at each temporal bin.")

    print()
    print("PRIME OPTICAL MODES")
    print("-" * 100)

    qiskit_interface = QiskitOpticalInterface(total_modes, num_qubits=qubits)

    for intensity, mode, temporal, spatial in result["top_prime_modes"][:top_k]:
        bits = qiskit_interface.report_mode(mode)
        row = spatial // spatial_cols
        col = spatial % spatial_cols

        print(
            f"mode={mode:12d} |{bits}> "
            f"prime={mode:<12d} temporal={temporal:<8d} "
            f"spatial={spatial:<8d} xy=({row:3d},{col:3d}) "
            f"optical={intensity:.8f}"
        )

    print()
    print(f"Total optical power : {result['total_power']:.8g}")
    print(f"Prime-mode power    : {result['prime_power']:.8g}")

    if result["total_power"] > 0:
        print(f"Prime fraction      : {result['prime_power'] / result['total_power']:.8%}")


# ===============================================================
# MAIN
# ===============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "General quantum computer via Qiskit + optical bench."
        )
    )

    # BACKEND
    parser.add_argument(
        "--backend",
        choices=["simulator", "optical"],
        default="optical",
        help="Execution backend: simulator (Aer) or optical (bench).",
    )

    # QUBITS / SHOTS
    parser.add_argument(
        "--qubits",
        type=int,
        default=2,
        help="Number of logical qubits for the quantum computer.",
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Number of shots / measurements.",
    )

    # CIRCUIT (simple operation list or script file)
    parser.add_argument(
        "--gate",
        action="append",
        dest="gates",
        metavar="OP",
        help=(
            "Gate in the form op:args, e.g. h:0, cx:0:1, rz:1.57079632679:1"
        ),
    )

    parser.add_argument(
        "--script",
        type=str,
        default=None,
        help="Path to a .q instruction file containing gate operations.",
    )

    # CAMERA
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic camera instead of physical camera.",
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

    # SPATIAL OPTICAL PLANE
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

    # TEMPORAL MULTIPLEXING
    parser.add_argument(
        "--modes-z",
        type=int,
        default=1,
        help="Number of temporal optical bins.",
    )

    # ROI
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
    )

    # ITO PATTERN
    parser.add_argument(
        "--pattern",
        choices=[
            "single", "checker", "row", "column",
            "random", "binary", "diagonal",
        ],
        default="single",
    )

    parser.add_argument(
        "--pattern-index",
        type=int,
        default=0,
    )

    # ITO
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

    # PRIME
    parser.add_argument(
        "--prime-only",
        action="store_true",
        help="Only retain prime-number optical addresses.",
    )

    # REPORT
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

    # VALIDATION
    if args.qubits <= 0:
        raise ValueError("--qubits must be > 0")
    if args.modes_x <= 0:
        raise ValueError("--modes-x must be > 0")
    if args.modes_y <= 0:
        raise ValueError("--modes-y must be > 0")
    if args.modes_z <= 0:
        raise ValueError("--modes-z must be > 0")

    # BUILD CIRCUIT
    operation_list = []
    if args.script:
        operation_list.extend(parse_q_file(args.script))
    if args.gates:
        operation_list.extend(args.gates)

    if operation_list:
        qc = parse_operations(operation_list, args.qubits)
    else:
        qc = None  # let GeneralQuantumComputer use default

    # BACKEND SELECTION
    if args.backend == "simulator":
        computer = GeneralQuantumComputer(
            num_qubits=args.qubits,
            optical_backend=None,
            seed=2026,
        )

        if qc is None:
            qc = computer.build_circuit()

        result = computer.run_simulator(qc, shots=args.shots)

        print()
        print("=" * 100)
        print("SIMULATOR RESULT")
        print("=" * 100)
        print(f"Qubits : {args.qubits}")
        print(f"Shots  : {args.shots}")
        print("Counts :")
        for bitstring, count in sorted(result.counts.items()):
            print(f"  |{bitstring}> : {count}")

        return

    # args.backend == "optical"
    # CAMERA
    if args.synthetic:
        camera = SyntheticCamera(
            width=args.camera_width,
            height=args.camera_height,
            noise=args.synthetic_noise,
            seed=args.synthetic_seed,
            structured=True,
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

    # ITO
    ito = ITOController(
        port=args.serial,
        baudrate=args.baudrate,
        settle_time=args.settle,
    )

    # OPTICAL BENCH
    bench = OpticalBench(
        camera=camera,
        rows=args.modes_y,
        cols=args.modes_x,
        roi=(tuple(args.roi) if args.roi else None),
    )

    try:
        optical_backend = OpticalBackend(
            bench=bench,
            ito=ito,
            num_qubits=args.qubits,
            spatial_rows=args.modes_y,
            spatial_cols=args.modes_x,
            temporal_bins=args.modes_z,
            pattern_type=args.pattern,
            prime_only=args.prime_only,
        )

        computer = GeneralQuantumComputer(
            num_qubits=args.qubits,
            optical_backend=optical_backend,
            seed=2026,
        )

        if qc is None:
            qc = computer.build_circuit()

        result = computer.run_optical(qc, shots=args.shots)

        print()
        print("=" * 100)
        print("OPTICAL BACKEND RESULT")
        print("=" * 100)
        print(f"Qubits : {args.qubits}")
        print(f"Shots  : {args.shots}")
        print("Counts :")
        for bitstring, count in sorted(
            result.counts.items(), key=lambda item: item[1], reverse=True
        ):
            print(f"  |{bitstring}> : {count}")

    finally:
        camera.close()
        ito.close()


if __name__ == "__main__":
    main()
