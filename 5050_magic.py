#!/usr/bin/env python3
"""
real_optical_bench_arrival_compare.py

Real optical bench + ITO spatial modulator + streamed temporal bins.
Compares camera-derived optical arrival proxies after an unblock command
for deterministic and random instruction schedules.

Important measurement note
--------------------------
A normal USB/CMOS camera does not time-tag individual photons. This program
therefore records a frame timestamp and treats a thresholded optical response
in each camera-derived spatial mode as an arrival proxy. For single-photon
arrival-time measurements, replace or supplement OpticalCamera with a hardware
TDC/SPAD/TCSPC backend that yields event timestamps.

The program never materializes a [schedule, temporal_bin, spatial_mode] cube.
It processes one frame and one spatial plane at a time.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

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

    def _instruction_index(self, schedule: str, trial: int, temporal_bin: int):
        deterministic = self.base_pattern_index + trial * self.temporal_bins + temporal_bin
        if schedule == "deterministic":
            return deterministic
        span = max(1, int(abs(self.pressure_strength) * 1_000_003))
        return deterministic + int(self.rng.integers(0, span))

    def _pattern(self, schedule: str, trial: int, temporal_bin: int):
        index = self._instruction_index(schedule, trial, temporal_bin)
        kind = self.deterministic_pattern if schedule == "deterministic" else self.random_pattern
        return ITOController.generate_pattern(self.rows, self.cols, kind, index, self.rng)

    def run_schedule(self, schedule: str, trials: int):
        events = []
        frame_records = []
        for trial in range(trials):
            command_time_ns = time.perf_counter_ns()
            for t in range(self.temporal_bins):
                pattern = self._pattern(schedule, trial, t)
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
                            trial=trial,
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
                frame_records.append((schedule, trial, t, command_time_ns, frame_time_ns, float(modes.sum()), int(active.sum())))
                del frame, modes, pattern
        return events, frame_records


# ==============================================================
# REPORTING
# ==============================================================

def stats(events: Sequence[ArrivalEvent]):
    if not events:
        return {"count": 0, "mean_ns": None, "std_ns": None, "min_ns": None, "max_ns": None}
    x = np.asarray([e.delay_after_unblock_ns for e in events], dtype=np.float64)
    return {
        "count": int(x.size),
        "mean_ns": float(x.mean()),
        "std_ns": float(x.std()),
        "min_ns": float(x.min()),
        "max_ns": float(x.max()),
    }


def write_events(path: Path, events: Sequence[ArrivalEvent]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "schedule", "trial", "temporal_bin", "spatial_mode", "logical_mode",
            "command_time_ns", "frame_time_ns", "arrival_proxy_ns",
            "delay_after_unblock_ns", "intensity", "admitted",
        ])
        for e in events:
            w.writerow([
                e.schedule, e.trial, e.temporal_bin, e.spatial_mode, e.logical_mode,
                e.command_time_ns, e.frame_time_ns, e.arrival_proxy_ns,
                e.delay_after_unblock_ns, e.intensity, e.admitted,
            ])


def write_frames(path: Path, records):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["schedule", "trial", "temporal_bin", "command_time_ns", "frame_time_ns", "total_mode_intensity", "active_ito_cells"])
        w.writerows(records)


def main():
    p = argparse.ArgumentParser(description="Real optical-bench arrival-proxy comparison after ITO unblocking.")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic frames instead of a physical camera.")
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
    p.add_argument("--trials", type=int, default=8, help="Unblock trials for each schedule.")
    p.add_argument("--bin-period", type=float, default=0.050, help="Nominal temporal-bin period, seconds.")
    p.add_argument("--threshold", type=float, default=30.0, help="Camera-mode intensity threshold for an arrival proxy.")
    p.add_argument("--serial", default=None, help="ITO serial port, for example COM3 or /dev/ttyUSB0.")
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--settle", type=float, default=0.050, help="ITO settle time after each pattern, seconds.")
    p.add_argument("--deterministic-pattern", choices=["single", "checker", "row", "column", "diagonal", "binary", "open", "blocked"], default="single")
    p.add_argument("--random-pattern", choices=["random", "single", "checker", "row", "column", "diagonal", "binary", "open", "blocked"], default="random")
    p.add_argument("--pattern-index", type=int, default=0)
    p.add_argument("--pressure-strength", type=float, default=1.0, help="Range scaling for random instruction indices.")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", default="real_optical_arrival_output")
    args = p.parse_args()

    if args.modes_x <= 0 or args.modes_y <= 0 or args.modes_z <= 0 or args.trials <= 0:
        raise ValueError("modes-x, modes-y, modes-z, and trials must be > 0")
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
        print(f"UNBLOCK TRIALS / SCHEDULE : {args.trials}")
        print(f"ARRIVAL THRESHOLD         : {args.threshold}")
        print()

        deterministic_events, deterministic_frames = engine.run_schedule("deterministic", args.trials)
        random_events, random_frames = engine.run_schedule("random", args.trials)

        sd = stats(deterministic_events)
        sr = stats(random_events)
        print("ARRIVAL-PROXY DELAY AFTER UNBLOCK (ns)")
        print("-" * 72)
        print(f"deterministic: events={sd['count']}, mean={sd['mean_ns']}, std={sd['std_ns']}, min={sd['min_ns']}, max={sd['max_ns']}")
        print(f"random       : events={sr['count']}, mean={sr['mean_ns']}, std={sr['std_ns']}, min={sr['min_ns']}, max={sr['max_ns']}")

        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        write_events(out / "deterministic_arrival_events.csv", deterministic_events)
        write_events(out / "random_arrival_events.csv", random_events)
        write_frames(out / "deterministic_frame_records.csv", deterministic_frames)
        write_frames(out / "random_frame_records.csv", random_frames)
        metadata = {
            "camera_mode": camera_name,
            "serial_port": args.serial,
            "modes_x": args.modes_x,
            "modes_y": args.modes_y,
            "temporal_bins": args.modes_z,
            "trials_per_schedule": args.trials,
            "bin_period_s": args.bin_period,
            "threshold": args.threshold,
            "deterministic_pattern": args.deterministic_pattern,
            "random_pattern": args.random_pattern,
            "pressure_strength": args.pressure_strength,
            "deterministic_stats": sd,
            "random_stats": sr,
            "measurement_note": "Camera frame timing and threshold crossings are arrival proxies, not single-photon time tags.",
        }
        (out / "metadata.txt").write_text("\n".join(f"{k} = {v}" for k, v in metadata.items()), encoding="utf-8")
        print(f"\nSaved results to: {out}")
    finally:
        camera.close()
        ito.close()


if __name__ == "__main__":
    main()
