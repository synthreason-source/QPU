#!/usr/bin/env python3
"""
continuum_optical_bench_arrival_live.py

Real/synthetic optical bench + ITO spatial modulator + streamed temporal bins.
Compares photon arrival proxies after unblocking under:
  - deterministically sampled futures
  - randomly sampled futures
driven by an existent instruction pressure.

Continuum rule
--------------
Trials are run sequentially in a continuum with a finite instruction budget.
For each trial:
  - A predictor guesses "deterministic" vs "random" from the data.
  - If the prediction is correct:
      * The continuum lengthens (extra instructions added).
      * The instruction pressure is biased toward that schedule type.
  - If the prediction is incorrect:
      * No extra instructions are added.
      * The continuum continues only while instructions remain; otherwise it stops
        when the instruction budget runs out.

Live display
------------
Prints a real-time, updating line per trial and a running summary table.

Measurement note
----------------
Camera frame timing is an arrival proxy, not single-photon time tagging.
"""

from __future__ import annotations

import argparse
import csv
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, List

import cv2
import numpy as np

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


# ==============================================================
# CAMERA
# ==============================================================

class SyntheticCamera:
    def __init__(self, width=1920, height=1080, noise=0.03, seed=2026):
        self.width = int(width)
        self.height = int(height)
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed)
        self.frame_index = 0

    def read(self):
        h, w = self.height, self.width
        yy, xx = np.mgrid[0:h, 0:w]
        frame = self.rng.random((h, w), dtype=np.float64) * 0.10
        for _ in range(24):
            cx = self.rng.uniform(0, w)
            cy = self.rng.uniform(0, h)
            sx = self.rng.uniform(w * 0.01, w * 0.08)
            sy = self.rng.uniform(h * 0.01, h * 0.08)
            amp = self.rng.uniform(0.15, 1.0)
            frame += amp * np.exp(-((xx-cx)**2/(2*sx*sx) + (yy-cy)**2/(2*sy*sy)))
        frame *= 0.85 + 0.15 * np.sin(self.frame_index * 0.37)
        frame += self.rng.normal(0.0, self.noise, frame.shape)
        frame = np.clip(frame, 0.0, None)
        frame /= max(float(frame.max()), 1e-12)
        self.frame_index += 1
        return (frame * 255.0).astype(np.float64)

    def close(self):
        pass


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


# ==============================================================
# ITO CONTROLLER
# ==============================================================

class ITOController:
    def __init__(self, port: Optional[str], baudrate=115200, settle_time=0.05):
        self.port = port
        self.baudrate = int(baudrate)
        self.settle_time = float(settle_time)
        self.serial = None
        if port is not None:
            if not SERIAL_AVAILABLE:
                raise RuntimeError("pyserial is required for --serial: python -m pip install pyserial")
            self.serial = serial.Serial(port, baudrate=self.baudrate, timeout=1)
            time.sleep(0.5)

    @staticmethod
    def generate_pattern(rows, cols, pattern_type, index, rng):
        p = np.zeros((rows, cols), dtype=np.uint8)
        if pattern_type == "single":
            p[(index // cols) % rows, index % cols] = 1
        elif pattern_type == "checker":
            rr, cc = np.mgrid[:rows, :cols]
            p[:] = ((rr + cc + index) & 1).astype(np.uint8)
        elif pattern_type == "row":
            p[index % rows, :] = 1
        elif pattern_type == "column":
            p[:, index % cols] = 1
        elif pattern_type == "diagonal":
            offset = index % (rows + cols - 1)
            for r in range(rows):
                c = offset - r
                if 0 <= c < cols:
                    p[r, c] = 1
        elif pattern_type == "binary":
            for k in range(rows * cols):
                p[k // cols, k % cols] = (index >> k) & 1
        elif pattern_type == "random":
            p[:] = rng.integers(0, 2, size=(rows, cols), dtype=np.uint8)
        elif pattern_type == "open":
            p[:] = 1
        elif pattern_type == "blocked":
            p[:] = 0
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        return p

    def send_pattern(self, pattern):
        pattern = np.asarray(pattern, dtype=np.uint8)
        if self.serial is None:
            time.sleep(self.settle_time)
            return
        rows, cols = pattern.shape
        self.serial.write(f"BEGIN {rows} {cols}\n".encode())
        for row in pattern:
            self.serial.write(("".join("1" if x else "0" for x in row) + "\n").encode())
        self.serial.write(b"END\n")
        self.serial.flush()
        time.sleep(self.settle_time)

    def close(self):
        if self.serial is not None:
            self.serial.close()
            self.serial = None


# ==============================================================
# OPTICAL BENCH
# ==============================================================

class OpticalBench:
    def __init__(self, camera, rows, cols, roi=None):
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
        h, w = frame.shape
        modes = np.zeros((self.rows, self.cols), dtype=np.float64)
        for r in range(self.rows):
            y0, y1 = r * h // self.rows, (r + 1) * h // self.rows
            for c in range(self.cols):
                x0, x1 = c * w // self.cols, (c + 1) * w // self.cols
                region = frame[y0:y1, x0:x1]
                if region.size:
                    modes[r, c] = float(region.mean())
        return modes


# ==============================================================
# ARRIVAL-PROXY EXPERIMENT
# ==============================================================

@dataclass
class ArrivalEvent:
    schedule: str
    trial: int
    temporal_bin: int
    spatial_mode: int
    logical_mode: int
    command_time_ns: int
    frame_time_ns: int
    arrival_proxy_ns: int
    delay_after_unblock_ns: int
    intensity: float
    admitted: int


class RealOpticalArrivalComparison:
    def __init__(
        self,
        bench: OpticalBench,
        ito: ITOController,
        rows: int,
        cols: int,
        temporal_bins: int,
        bin_period_s: float,
        threshold: float,
        deterministic_pattern: str,
        random_pattern: str,
        base_pattern_index: int,
        pressure_strength: float,
        rng_seed: int,
    ):
        self.bench = bench
        self.ito = ito
        self.rows = int(rows)
        self.cols = int(cols)
        self.spatial_modes = self.rows * self.cols
        self.temporal_bins = int(temporal_bins)
        self.bin_period_s = float(bin_period_s)
        self.threshold = float(threshold)
        self.deterministic_pattern = deterministic_pattern
        self.random_pattern = random_pattern
        self.base_pattern_index = int(base_pattern_index)
        self.pressure_strength = float(pressure_strength)
        self.rng = np.random.default_rng(rng_seed)

    def _instruction_index(self, schedule: str, trial: int, temporal_bin: int, bias: float):
        deterministic = self.base_pattern_index + trial * self.temporal_bins + temporal_bin
        if schedule == "deterministic":
            return deterministic
        span = max(1, int(abs(self.pressure_strength * (1.0 + bias)) * 1_000_003))
        return deterministic + int(self.rng.integers(0, span))

    def _pattern(self, schedule: str, trial: int, temporal_bin: int, bias: float):
        index = self._instruction_index(schedule, trial, temporal_bin, bias)
        kind = self.deterministic_pattern if schedule == "deterministic" else self.random_pattern
        return ITOController.generate_pattern(self.rows, self.cols, kind, index, self.rng)

    def run_schedule(self, schedule: str, trial_offset: int, bias: float):
        events = []
        frame_records = []
        command_time_ns = time.perf_counter_ns()
        for t in range(self.temporal_bins):
            pattern = self._pattern(schedule, trial_offset, t, bias)
            self.ito.send_pattern(pattern)
            frame = self.bench.acquire()
            frame_time_ns = time.perf_counter_ns()
            modes = self.bench.measure_modes(frame).reshape(-1)
            expected_proxy_ns = command_time_ns + int((t + 1) * self.bin_period_s * 1e9)
            timestamp_ns = max(frame_time_ns, expected_proxy_ns)
            active = pattern.reshape(-1).astype(bool)
            for s, intensity in enumerate(modes):
                admitted = int(active[s] and intensity >= self.threshold)
                if admitted:
                    events.append(ArrivalEvent(
                        schedule=schedule,
                        trial=trial_offset,
                        temporal_bin=t,
                        spatial_mode=s,
                        logical_mode=t * self.spatial_modes + s,
                        command_time_ns=command_time_ns,
                        frame_time_ns=frame_time_ns,
                        arrival_proxy_ns=timestamp_ns,
                        delay_after_unblock_ns=timestamp_ns-command_time_ns,
                        intensity=float(intensity),
                        admitted=1,
                    ))
            frame_records.append((schedule, trial_offset, t, command_time_ns, frame_time_ns, float(modes.sum()), int(active.sum())))
            del frame, modes, pattern
        return events, frame_records


# ==============================================================
# CONTINUUM EXPERIMENT
# ==============================================================

@dataclass
class ContinuumTrial:
    index: int
    true_schedule: str
    event_count: int
    mean_delay_ns: float
    std_delay_ns: float
    mean_intensity: float
    active_ito_cells: int
    total_mode_intensity: float
    predicted_label: str
    confidence: float
    correct: bool
    instructions_remaining: int
    continued: bool


def summarize_one_schedule(events: Sequence[ArrivalEvent], frame_records) -> dict:
    if not events:
        return {
            "event_count": 0,
            "mean_delay_ns": 0.0,
            "std_delay_ns": 0.0,
            "mean_intensity": 0.0,
            "active_ito_cells": 0,
            "total_mode_intensity": 0.0,
        }
    delays = np.asarray([e.delay_after_unblock_ns for e in events], dtype=np.float64)
    intensities = np.asarray([e.intensity for e in events], dtype=np.float64)
    total_int = sum(r[5] for r in frame_records)
    active_cells = sum(r[6] for r in frame_records)
    return {
        "event_count": int(delays.size),
        "mean_delay_ns": float(delays.mean()) if delays.size else 0.0,
        "std_delay_ns": float(delays.std()) if delays.size else 0.0,
        "mean_intensity": float(intensities.mean()) if intensities.size else 0.0,
        "active_ito_cells": int(active_cells),
        "total_mode_intensity": float(total_int),
    }


def predict_schedule(stats: dict, rng: np.random.Generator) -> tuple[str, float]:
    rel_std = stats["std_delay_ns"] / max(stats["mean_delay_ns"], 1.0)
    count_norm = min(1.0, stats["event_count"] / 64.0)
    relstd_score = 1.0 / (1.0 + np.exp(5.0 * (rel_std - 0.5)))
    score = 0.6 * count_norm + 0.4 * relstd_score
    p_det = 1.0 / (1.0 + np.exp(-4.0 * (score - 0.5)))
    label = "deterministic" if rng.random() < p_det else "random"
    conf = float(max(p_det, 1.0 - p_det))
    return label, conf


def run_continuum_live(
    engine: RealOpticalArrivalComparison,
    base_instructions: int,
    rng: np.random.Generator,
    live_delay: float = 0.0,
):
    """
    Generator that yields (trial, running_summary) in real time.
    """
    instructions = base_instructions
    bias = 0.0
    trial_counter = 0
    history = []

    while instructions > 0:
        true_schedule = "deterministic" if rng.random() < 0.5 else "random"
        events, frames = engine.run_schedule(true_schedule, trial_offset=trial_counter, bias=bias)
        stats = summarize_one_schedule(events, frames)
        pred_label, conf = predict_schedule(stats, rng)
        correct = (pred_label == true_schedule)

        instructions -= 1
        if correct:
            instructions += 2
            if true_schedule == "deterministic":
                bias = min(5.0, bias + 0.5)
            else:
                bias = max(-5.0, bias - 0.5)

        continued = (instructions > 0)

        ct = ContinuumTrial(
            index=trial_counter,
            true_schedule=true_schedule,
            event_count=stats["event_count"],
            mean_delay_ns=stats["mean_delay_ns"],
            std_delay_ns=stats["std_delay_ns"],
            mean_intensity=stats["mean_intensity"],
            active_ito_cells=stats["active_ito_cells"],
            total_mode_intensity=stats["total_mode_intensity"],
            predicted_label=pred_label,
            confidence=conf,
            correct=correct,
            instructions_remaining=instructions,
            continued=continued,
        )
        history.append(ct)

        yield ct, list(history)

        trial_counter += 1
        if not continued:
            break

        if live_delay > 0:
            time.sleep(live_delay)


# ==============================================================
# LIVE DISPLAY
# ==============================================================

def print_header():
    print("=" * 90)
    print("CONTINUUM OPTICAL BENCH — LIVE")
    print("=" * 90)
    print(f"{'trial':>5} | {'true':>12} | {'pred':>12} | {'correct':>7} | {'conf':>5} | {'instr_left':>10} | {'status':>8}")
    print("-" * 90)
    sys.stdout.flush()


def print_trial_row(ct: ContinuumTrial):
    status = "CONTINUE" if ct.continued else "STOP"
    line = (
        f"{ct.index:5d} | {ct.true_schedule:>12} | {ct.predicted_label:>12} | "
        f"{int(ct.correct):>7} | {ct.confidence:5.3f} | {ct.instructions_remaining:10d} | {status:>8}"
    )
    print(line)
    sys.stdout.flush()


def print_summary(history: list[ContinuumTrial]):
    if not history:
        return
    n = len(history)
    correct_count = sum(1 for h in history if h.correct)
    det_count = sum(1 for h in history if h.true_schedule == "deterministic")
    ran_count = n - det_count
    det_correct = sum(1 for h in history if h.true_schedule == "deterministic" and h.correct)
    ran_correct = sum(1 for h in history if h.true_schedule == "random" and h.correct)
    avg_conf = np.mean([h.confidence for h in history])
    final_instr = history[-1].instructions_remaining

    print()
    print("SUMMARY")
    print("-" * 90)
    print(f"trials so far         : {n}")
    print(f"correct predictions   : {correct_count} / {n}  ({correct_count/n*100:.1f}%)")
    print(f"deterministic trials  : {det_count}  (correct {det_correct})")
    print(f"random trials         : {ran_count}  (correct {ran_correct})")
    print(f"avg confidence        : {avg_conf:.3f}")
    print(f"instructions remaining: {final_instr}")
    print("=" * 90)
    sys.stdout.flush()


# ==============================================================
# CSV OUTPUT
# ==============================================================

def write_continuum(path: Path, trials: list[ContinuumTrial]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "index", "true_schedule", "event_count", "mean_delay_ns", "std_delay_ns",
            "mean_intensity", "active_ito_cells", "total_mode_intensity",
            "predicted_label", "confidence", "correct", "instructions_remaining", "continued",
        ])
        for t in trials:
            w.writerow([
                t.index, t.true_schedule, t.event_count, t.mean_delay_ns, t.std_delay_ns,
                t.mean_intensity, t.active_ito_cells, t.total_mode_intensity,
                t.predicted_label, t.confidence, int(t.correct), t.instructions_remaining, int(t.continued),
            ])


# ==============================================================
# MAIN
# ==============================================================

def main():
    p = argparse.ArgumentParser(description="Continuum optical-bench arrival comparison with live display.")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic frames.")
    p.add_argument("--synthetic-seed", type=int, default=2026)
    p.add_argument("--synthetic-noise", type=float, default=0.03)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--camera-width", type=int, default=1920)
    p.add_argument("--camera-height", type=int, default=1080)
    p.add_argument("--exposure", type=float, default=None)
    p.add_argument("--gain", type=float, default=None)
    p.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    p.add_argument("--modes-x", type=int, default=4)
    p.add_argument("--modes-y", type=int, default=4)
    p.add_argument("--modes-z", type=int, default=1, help="Temporal bins per unblock trial.")
    p.add_argument("--base-instructions", type=int, default=10, help="Initial instruction budget.")
    p.add_argument("--bin-period", type=float, default=0.050, help="Nominal temporal-bin period, seconds.")
    p.add_argument("--threshold", type=float, default=30.0, help="Intensity threshold for arrival proxy.")
    p.add_argument("--serial", default=None, help="ITO serial port.")
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--settle", type=float, default=0.050, help="ITO settle time, seconds.")
    p.add_argument("--deterministic-pattern", choices=["single", "checker", "row", "column", "diagonal", "binary", "open", "blocked"], default="single")
    p.add_argument("--random-pattern", choices=["random", "single", "checker", "row", "column", "diagonal", "binary", "open", "blocked"], default="random")
    p.add_argument("--pattern-index", type=int, default=0)
    p.add_argument("--pressure-strength", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", default="continuum_optical_output")
    p.add_argument("--live-delay", type=float, default=0.0, help="Artificial delay between trials for live demo.")
    args = p.parse_args()

    if args.modes_x <= 0 or args.modes_y <= 0 or args.modes_z <= 0 or args.base_instructions <= 0:
        raise ValueError("modes-x, modes-y, modes-z, and base-instructions must be > 0")
    if args.bin_period <= 0:
        raise ValueError("--bin-period must be > 0")

    if args.synthetic:
        camera = SyntheticCamera(args.camera_width, args.camera_height, args.synthetic_noise, args.synthetic_seed)
        camera_name = "synthetic"
    else:
        camera = OpticalCamera(CameraConfig(args.camera, args.camera_width, args.camera_height, args.exposure, args.gain))
        camera_name = "real"

    ito = ITOController(args.serial, args.baudrate, args.settle)
    bench = OpticalBench(camera, args.modes_y, args.modes_x, tuple(args.roi) if args.roi else None)
    engine = RealOpticalArrivalComparison(
        bench=bench,
        ito=ito,
        rows=args.modes_y,
        cols=args.modes_x,
        temporal_bins=args.modes_z,
        bin_period_s=args.bin_period,
        threshold=args.threshold,
        deterministic_pattern=args.deterministic_pattern,
        random_pattern=args.random_pattern,
        base_pattern_index=args.pattern_index,
        pressure_strength=args.pressure_strength,
        rng_seed=args.seed,
    )

    try:
        print(f"CAMERA MODE              : {camera_name.upper()}")
        print(f"ITO SERIAL PORT           : {args.serial if args.serial else 'none (timed dry-run)'}")
        print(f"SPATIAL MODES             : {args.modes_y} x {args.modes_x} = {args.modes_y * args.modes_x}")
        print(f"TEMPORAL BINS / TRIAL     : {args.modes_z}")
        print(f"BASE INSTRUCTIONS         : {args.base_instructions}")
        print(f"ARRIVAL THRESHOLD         : {args.threshold}")
        print()

        cont_rng = np.random.default_rng(args.seed + 777)
        print_header()

        all_trials = []
        for ct, history in run_continuum_live(engine, args.base_instructions, cont_rng, live_delay=args.live_delay):
            print_trial_row(ct)
            all_trials = history
            print_summary(history)

        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        write_continuum(out / "continuum_trials.csv", all_trials)

        metadata = {
            "camera_mode": camera_name,
            "serial_port": args.serial,
            "modes_x": args.modes_x,
            "modes_y": args.modes_y,
            "temporal_bins": args.modes_z,
            "base_instructions": args.base_instructions,
            "bin_period_s": args.bin_period,
            "threshold": args.threshold,
            "deterministic_pattern": args.deterministic_pattern,
            "random_pattern": args.random_pattern,
            "pressure_strength": args.pressure_strength,
            "continuum_length": len(all_trials),
            "final_trial_index": all_trials[-1].index if all_trials else -1,
            "final_instructions_remaining": all_trials[-1].instructions_remaining if all_trials else 0,
            "measurement_note": "Camera frame timing and threshold crossings are arrival proxies.",
            "continuum_rule": "Correct predictions add instructions and bias the continuum; incorrect predictions let instructions run out.",
        }
        (out / "metadata.txt").write_text("\n".join(f"{k} = {v}" for k, v in metadata.items()), encoding="utf-8")
        print(f"\nSaved results to: {out}")
    finally:
        camera.close()
        ito.close()


if __name__ == "__main__":
    main()
