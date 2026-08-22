from __future__ import annotations

"""Hybrid optical-bench/OpenCV -> Qiskit demonstration.

Synthetic dry run:
    python optical_bench_qiskit.py --demo --synthetic --quantum-demo

Live camera example:
    python optical_bench_qiskit.py --demo --camera 0 --roi 100 80 900 600 --quantum-demo

The physical plate or SLM must be set to the intended A/B pattern before
capture.  This program reads the camera, decodes an output grid, and sends
the decoded non-negative intensity matrix into a real Qiskit circuit.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
import argparse
import math
import time

import cv2
import numpy as np

try:
    import qiskit  # noqa: F401
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


@dataclass
class CameraConfig:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    exposure: Optional[float] = None
    gain: Optional[float] = None
    warmup_frames: int = 12


class OpenCVCamera:
    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap = cv2.VideoCapture(config.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {config.camera_index}. "
                "Use --synthetic or select the correct --camera index."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        if config.exposure is not None:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, config.exposure)
        if config.gain is not None:
            self.cap.set(cv2.CAP_PROP_GAIN, config.gain)
        for _ in range(max(0, config.warmup_frames)):
            self.read()

    def read(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Camera frame acquisition failed")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def close(self) -> None:
        if getattr(self, "cap", None) is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class SyntheticCamera:
    """Dry-run camera whose grid encodes a supplied optical-MAC matrix."""
    def __init__(self, frame_shape=(720, 1280), noise_std=1.5, seed=2026):
        self.frame_shape = tuple(frame_shape)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)
        self.matrix: Optional[np.ndarray] = None

    def set_matrix(self, matrix: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float32)

    def read(self) -> np.ndarray:
        if self.matrix is None:
            raise RuntimeError("SyntheticCamera has no optical matrix loaded")
        rows, cols = self.matrix.shape
        h, w = self.frame_shape
        frame = np.full((h, w), 12.0, dtype=np.float32)
        peak = max(float(self.matrix.max()), 1.0)
        for r in range(rows):
            y0, y1 = round(r * h / rows), round((r + 1) * h / rows)
            for c in range(cols):
                x0, x1 = round(c * w / cols), round((c + 1) * w / cols)
                intensity = 20.0 + 220.0 * self.matrix[r, c] / peak
                frame[y0:y1, x0:x1] = intensity
        frame += self.rng.normal(0.0, self.noise_std, frame.shape)
        return np.clip(frame, 0, 255).astype(np.uint8)

    def close(self) -> None:
        pass


class OpticalBench:
    def __init__(
        self,
        camera: Any,
        output_shape: Sequence[int],
        roi: Optional[Sequence[int]] = None,
        background: Optional[np.ndarray] = None,
        flat_field: Optional[np.ndarray] = None,
        calibration: Optional[tuple[float, float]] = None,
    ):
        self.camera = camera
        self.output_shape = tuple(map(int, output_shape))
        if len(self.output_shape) != 2 or min(self.output_shape) <= 0:
            raise ValueError("output_shape must be (positive_rows, positive_cols)")
        self.roi = tuple(map(int, roi)) if roi is not None else None
        self.background = background
        self.flat_field = flat_field
        self.calibration = calibration

    @staticmethod
    def load_image(path: str | Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not load calibration image: {path}")
        return image.astype(np.float32)

    @staticmethod
    def _matching_calibration(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)
        if image.shape != shape:
            image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        return image

    def capture(self) -> np.ndarray:
        frame = self.camera.read().astype(np.float32)
        if self.roi is not None:
            x, y, w, h = self.roi
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                raise ValueError("ROI must be (x, y, width, height), all in-frame")
            frame = frame[y:y + h, x:x + w]
            if frame.size == 0:
                raise ValueError("ROI is outside the camera frame")
        if self.background is not None:
            frame -= self._matching_calibration(self.background, frame.shape)
        if self.flat_field is not None:
            frame /= np.maximum(self._matching_calibration(self.flat_field, frame.shape), 1.0)
        return np.maximum(frame, 0.0)

    def decode_cells(self, frame: np.ndarray) -> np.ndarray:
        rows, cols = self.output_shape
        h, w = frame.shape
        decoded = np.zeros((rows, cols), dtype=np.float32)
        for r in range(rows):
            y0, y1 = round(r * h / rows), round((r + 1) * h / rows)
            for c in range(cols):
                x0, x1 = round(c * w / cols), round((c + 1) * w / cols)
                cell = frame[y0:y1, x0:x1]
                decoded[r, c] = float(cell.mean()) if cell.size else 0.0
        return decoded

    def fit_linear_calibration(self, raw: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
        x = np.asarray(raw, dtype=np.float64).ravel()
        y = np.asarray(expected, dtype=np.float64).ravel()
        if x.size != y.size or x.size < 2:
            raise ValueError("raw and expected need matching sizes of at least two")
        alpha, beta = np.polyfit(x, y, 1)
        self.calibration = (float(alpha), float(beta))
        return self.calibration

    def apply_calibration(self, raw: np.ndarray, max_count: int) -> np.ndarray:
        if self.calibration is None:
            lo, hi = float(raw.min()), float(raw.max())
            if hi <= lo:
                output = np.zeros_like(raw)
            else:
                output = (raw - lo) * max_count / (hi - lo)
        else:
            alpha, beta = self.calibration
            output = alpha * raw + beta
        return np.clip(np.rint(output), 0, max_count).astype(np.float32)

    def run_mac(self, A: np.ndarray, B: np.ndarray) -> dict[str, np.ndarray]:
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("A and B must each be two-dimensional")
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Incompatible matrix shapes {A.shape} @ {B.shape}")
        if self.output_shape != (A.shape[0], B.shape[1]):
            raise ValueError(f"output_shape must be {(A.shape[0], B.shape[1])}")
        if not np.isin(A, [0, 1]).all() or not np.isin(B, [0, 1]).all():
            raise ValueError("The plate MAC example accepts only binary A and B")

        reference = A @ B
        if isinstance(self.camera, SyntheticCamera):
            self.camera.set_matrix(reference)

        frame = self.capture()
        raw = self.decode_cells(frame)
        estimate = self.apply_calibration(raw, max_count=A.shape[1])
        return {
            "A": A,
            "B": B,
            "camera_frame": frame,
            "optical_raw": raw,
            "optical_estimate": estimate,
            "digital_reference": reference,
            "absolute_error": np.abs(estimate - reference),
        }


class OpticalJob:
    """Completed synchronous job object; avoids Qiskit provider-version ABCs."""
    def __init__(self, backend: Any, job_id: str, payload: dict):
        self._backend = backend
        self._job_id = job_id
        self._payload = payload

    def backend(self):
        return self._backend

    def job_id(self):
        return self._job_id

    def submit(self):
        return None

    def status(self):
        return "DONE"

    def result(self, timeout: Optional[float] = None):
        return self._payload

    def done(self):
        return True


class OpticalMACBackend:
    """Qiskit-style wrapper around camera-in-the-loop optical MAC capture."""
    def __init__(self, bench: OpticalBench, name="optical_mac_backend"):
        self.bench = bench
        self.name = name

    def run(self, run_input: dict, **kwargs) -> OpticalJob:
        payload = run_input[0] if isinstance(run_input, (tuple, list)) else run_input
        if not isinstance(payload, dict) or "A" not in payload or "B" not in payload:
            raise ValueError("Expected {'A': binary_matrix, 'B': binary_matrix}")
        data = self.bench.run_mac(payload["A"], payload["B"])
        job_id = f"optical-{time.time_ns()}"
        return OpticalJob(self, job_id, {
            "backend_name": self.name,
            "job_id": job_id,
            "success": True,
            "data": data,
        })


def optical_matrix_to_amplitudes(optical_matrix: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Map a non-negative optical output grid to a normalized quantum state."""
    values = np.maximum(np.asarray(optical_matrix, dtype=np.float64).ravel(), 0.0)
    if values.size == 0:
        raise ValueError("Cannot encode an empty optical matrix")
    if np.isclose(values.sum(), 0.0):
        values = np.ones_like(values)
    probabilities = values / values.sum()
    data_cells = values.size
    n_qubits = max(1, math.ceil(math.log2(data_cells)))
    amplitudes = np.zeros(1 << n_qubits, dtype=np.complex128)
    amplitudes[:data_cells] = np.sqrt(probabilities)
    return amplitudes, n_qubits, data_cells


def build_qiskit_optical_circuit(optical_matrix: np.ndarray, entangle: bool = True):
    """Camera matrix -> amplitude initialization -> Qiskit gates -> measurements."""
    if not QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit is not installed. Run: pip install qiskit")
    from qiskit import QuantumCircuit

    amplitudes, n_qubits, data_cells = optical_matrix_to_amplitudes(optical_matrix)
    circuit = QuantumCircuit(n_qubits, n_qubits, name="camera_optical_qiskit")
    circuit.initialize(amplitudes, range(n_qubits))

    if entangle and n_qubits > 1:
        for q in range(n_qubits):
            circuit.ry(np.pi / 4, q)
        for q in range(n_qubits - 1):
            circuit.cx(q, q + 1)
        for q in range(n_qubits - 1, 0, -1):
            circuit.cx(q, q - 1)

    circuit.measure(range(n_qubits), range(n_qubits))
    return circuit, {
        "input_shape": tuple(np.asarray(optical_matrix).shape),
        "data_cells": data_cells,
        "num_qubits": n_qubits,
        "amplitudes": amplitudes,
    }

def run_via_qiskit(
    circuit,
    shots: int = 8192,
    seed: int = 1234,
) -> dict:
    """
    Execute the measured circuit with Qiskit's V2 statevector sampler.

    Requires:
        pip install -U qiskit
    """
    if not QISKIT_AVAILABLE:
        raise RuntimeError(
            "Qiskit is not installed. Run: pip install -U qiskit"
        )

    from qiskit.primitives import StatevectorSampler

    sampler = StatevectorSampler(
        default_shots=shots,
        seed=seed,
    )

    job = sampler.run(
        [circuit],
        shots=shots,
    )

    primitive_result = job.result()
    pub_result = primitive_result[0]

    # "meas" is the classical register created by:
    # QuantumCircuit(n_qubits, n_qubits)
    # Works regardless of whether Qiskit called the register "c", "meas",
    # or another explicit ClassicalRegister name.
    register_names = list(pub_result.data)

    if not register_names:
        raise RuntimeError(
            "Qiskit returned no classical-register measurement data. "
            "Ensure the circuit ends with measure(...)."
        )

    if len(register_names) == 1:
        register_data = pub_result.data[register_names[0]]
        counts = dict(register_data.get_counts())
    else:
    # Join all classical registers before extracting counts.
        counts = dict(pub_result.join_data().get_counts())
    return {
        "engine": "qiskit.primitives.StatevectorSampler",
        "shots": shots,
        "counts": counts,
    }


def decode_quantum_counts(counts: dict[str, int], optical_matrix: np.ndarray, top_k: int = 16) -> list[dict]:
    """Relate measured basis labels to flattened optical matrix cells."""
    rows, cols = np.asarray(optical_matrix).shape
    n_cells = rows * cols
    total = max(sum(counts.values()), 1)
    decoded = []
    for bits, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:top_k]:
        compact = bits.replace(" ", "")
        index = int(compact[::-1], 2)  # displayed clbit order -> initialize basis index
        entry = {"bitstring": compact, "basis_index": index, "count": count, "probability": count / total}
        if index < n_cells:
            row, col = divmod(index, cols)
            entry.update({"row": row, "column": col, "optical_mac_value": float(optical_matrix[row, col])})
        else:
            entry.update({"row": None, "column": None, "optical_mac_value": None})
        decoded.append(entry)
    return decoded


def save_diagnostics(data: dict, out_dir: str | Path = "optical_output") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "camera_frame.png"), np.clip(data["camera_frame"], 0, 255).astype(np.uint8))
    for key in ("optical_raw", "optical_estimate", "digital_reference", "absolute_error"):
        np.savetxt(out / f"{key}.csv", data[key], delimiter=",", fmt="%.6g")


def save_quantum_output(quantum: dict, out_dir: str | Path = "optical_output") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "qiskit_optical_circuit.txt").write_text(str(quantum["circuit"].draw("text")), encoding="utf-8")
    np.savetxt(out / "quantum_input_optical_matrix.csv", quantum["optical_matrix"], delimiter=",", fmt="%.6g")
    with (out / "qiskit_measurement_counts.csv").open("w", encoding="utf-8") as f:
        f.write("bitstring,count\n")
        for bits, count in sorted(quantum["sampler"]["counts"].items(), key=lambda pair: pair[1], reverse=True):
            f.write(f"{bits},{count}\n")


def run_quantum_demo(optical_result: dict, shots: int) -> dict:
    matrix = optical_result["optical_estimate"]
    circuit, metadata = build_qiskit_optical_circuit(matrix, entangle=True)
    sampler = run_via_qiskit(circuit, shots=shots)
    return {
        "optical_matrix": matrix,
        "circuit": circuit,
        "encoding": metadata,
        "sampler": sampler,
        "states": decode_quantum_counts(sampler["counts"], matrix),
    }


def demo(synthetic: bool, camera_index: int, roi: Optional[Sequence[int]], quantum_demo: bool, shots: int) -> None:
    A = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 1], [0, 0, 1, 1]], dtype=np.float32)
    B = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32)
    camera = SyntheticCamera() if synthetic else OpenCVCamera(CameraConfig(camera_index=camera_index))
    try:
        bench = OpticalBench(camera=camera, output_shape=(A.shape[0], B.shape[1]), roi=roi)
        optical_backend = OpticalMACBackend(bench)
        result = optical_backend.run({"A": A, "B": B}).result()["data"]
        print("\n=== DIGITAL BINARY MAC REFERENCE ===\n", result["digital_reference"])
        print("\n=== CAMERA / OPTICAL MAC ESTIMATE ===\n", result["optical_estimate"])
        print("\n=== ABSOLUTE ERROR ===\n", result["absolute_error"])
        print("\nMaximum optical-camera error:", float(result["absolute_error"].max()))
        save_diagnostics(result)

        if quantum_demo:
            quantum = run_quantum_demo(result, shots=shots)
            print("\n=== QISKIT CIRCUIT ===")
            print(f"Encoded optical grid: {quantum['encoding']['input_shape']}")
            print(f"Data cells: {quantum['encoding']['data_cells']}; qubits: {quantum['encoding']['num_qubits']}")
            print(quantum["circuit"].draw("text"))
            print("\n=== QISKIT EXECUTION ===")
            print("Engine:", quantum["sampler"]["engine"])
            print("Shots:", quantum["sampler"]["shots"])
            print("\n=== TOP MEASURED BASIS STATES ===")
            for item in quantum["states"]:
                location = "padding" if item["row"] is None else f"cell=({item['row']},{item['column']}), optical_MAC={item['optical_mac_value']:.0f}"
                print(f"|{item['bitstring']}>  index={item['basis_index']:2d}  count={item['count']:5d}  p={item['probability']:.5f}  {location}")
            save_quantum_output(quantum)
            print("\nSaved optical and Qiskit outputs to optical_output")
    finally:
        camera.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenCV optical MAC and Qiskit hybrid demo")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Use a simulated camera instead of physical hardware")
    parser.add_argument("--quantum-demo", action="store_true", help="Encode camera matrix and run it with Qiskit")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--shots", type=int, default=8192)
    args = parser.parse_args()
    if not args.demo:
        parser.print_help()
        return
    demo(args.synthetic, args.camera, args.roi, args.quantum_demo, args.shots)


if __name__ == "__main__":
    main()
