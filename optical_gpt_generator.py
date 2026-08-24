#!/usr/bin/env python3
"""
GPT generation with optical-bench probability sampling.

This script:
- Loads a Hugging Face causal LM (e.g. GPT-2).
- At each generation step, sends the token probability distribution
  to an optical bench (ITO + camera) via OpticalProbabilitySampler.
- Uses the measured optical distribution to sample the next token.

Architecture:
    GPT forward pass -> logits -> probs
         |
         v
    encode probs onto ITO plane (streamed if needed)
         |
         v
    camera acquires optical frame(s)
         |
         v
    decode measured intensities -> optical probs
         |
         v
    sample next token from optical (or mixed) distribution
         |
         +--> append token and repeat

Requirements:
    pip install torch transformers opencv-python numpy

Optional (for real hardware):
    pip install pyserial
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# ===============================================================
# OPTIONAL SERIAL (ITO CONTROLLER)
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
    Synthetic camera for testing without real hardware.
    Generates structured random frames.
    """

    def __init__(
        self,
        width=640,
        height=480,
        noise=0.02,
        seed=2026,
        structured=True,
    ):
        self.width = int(width)
        self.height = int(height)
        self.noise = float(noise)
        self.structured = bool(structured)
        self.rng = np.random.default_rng(seed)
        self.frame_index = 0

    def read(self):
        h = self.height
        w = self.width

        field = self.rng.random((h, w), dtype=np.float64)

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
                    -((xx - cx) ** 2 / (2 * sx * sx)
                      + (yy - cy) ** 2 / (2 * sy * sy))
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
    width: int = 640
    height: int = 480
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
        self.settle_time = float(settle_time)
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
            print("ITO: no serial controller attached (simulation mode).")
            time.sleep(self.settle_time)
            return  # CRITICAL: must return here

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
# OPTICAL PROBABILITY SAMPLER
# ===============================================================

class OpticalProbabilitySampler:
    """
    Sends a probability vector to the optical bench in streamed chunks.

    The optical bench is used for measurement/sampling. It does not
    replace the language-model forward pass.
    """

    def __init__(
        self,
        bench,
        ito,
        rows,
        cols,
        temporal_bins=1,
        scale=255.0,
        settle_time=0.05,
        background=None,
    ):
        self.bench = bench
        self.ito = ito
        self.rows = int(rows)
        self.cols = int(cols)
        self.temporal_bins = int(temporal_bins)
        self.spatial_modes = self.rows * self.cols
        self.capacity = self.spatial_modes * self.temporal_bins
        self.scale = float(scale)
        self.settle_time = float(settle_time)
        self.background = (
            np.asarray(background, dtype=np.float64)
            if background is not None else None
        )

    def _normalise(self, probabilities):
        p = np.asarray(probabilities, dtype=np.float64)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        p = np.maximum(p, 0.0)
        total = p.sum()
        if total <= 0:
            raise ValueError("Probability vector has zero mass")
        return p / total

    def _probability_pattern(self, values):
        """
        Convert a probability slice into an 8-bit non-negative ITO pattern.
        """
        values = np.asarray(values, dtype=np.float64)
        values = np.clip(values, 0.0, 1.0)
        pattern = np.rint(values * self.scale)
        return pattern.reshape(self.rows, self.cols).astype(np.uint8)

    def measure_distribution(self, probabilities):
        """
        Load probabilities into the ITO plane, acquire optical frames,
        and decode the measured distribution.
        """
        p = self._normalise(probabilities)
        output = np.zeros_like(p)

        for start in range(0, len(p), self.spatial_modes):
            stop = min(start + self.spatial_modes, len(p))
            chunk = p[start:stop]

            padded = np.zeros(self.spatial_modes, dtype=np.float64)
            padded[:len(chunk)] = chunk

            pattern = self._probability_pattern(padded)
            self.ito.send_pattern(pattern)

            frame = self.bench.acquire()
            modes = self.bench.measure_modes(frame).reshape(-1)

            if self.background is not None:
                modes = modes - self.background

            modes = np.maximum(modes, 0.0)
            output[start:stop] = modes[:len(chunk)]

        total = output.sum()
        if total <= 0:
            return p  # fallback if optical signal is unusable

        return output / total

    def sample(self, probabilities, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        measured = self.measure_distribution(probabilities)
        token_id = int(rng.choice(len(measured), p=measured))
        return token_id, measured

# ===============================================================
# OPTICAL GPT GENERATOR
# ===============================================================

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

if TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE:

    class OpticalGPTGenerator:
        """
        Generates text from a causal LM, sampling next tokens using
        an optical probability distribution.
        """

        def __init__(
            self,
            model,
            tokenizer,
            optical_sampler,
            device="cuda",
            temperature=1.0,
            top_k=50,
            optical_mix=1.0,
            seed=2026,
            use_topk_optical=True,
            optical_topk=64,
        ):
            self.model = model.to(device).eval()
            self.tokenizer = tokenizer
            self.optical_sampler = optical_sampler
            self.device = device
            self.temperature = float(temperature)
            self.top_k = int(top_k)
            self.optical_mix = float(optical_mix)
            self.rng = np.random.default_rng(seed)
            self.use_topk_optical = bool(use_topk_optical)
            self.optical_topk = int(optical_topk)

        def _optical_topk_sample(
            self,
            digital_probs,
        ):
            """
            Select top-K digital candidates, load them onto the optical
            bench, and sample from the measured optical distribution.
            Returns (token_id, measured_probs, candidate_ids).
            """
            candidate_ids = np.argpartition(
                digital_probs, -self.optical_topk
            )[-self.optical_topk:]

            candidate_probs = digital_probs[candidate_ids]
            candidate_probs = np.maximum(candidate_probs, 0.0)
            total = candidate_probs.sum()
            if total <= 0:
                # Fallback to digital sampling over full vocab
                return (
                    int(self.rng.choice(len(digital_probs), p=digital_probs)),
                    digital_probs,
                    None,
                )

            candidate_probs /= total

            measured = self.optical_sampler.measure_distribution(candidate_probs)

            # Sample from measured optical distribution
            selected = int(self.rng.choice(len(candidate_ids), p=measured))
            token_id = int(candidate_ids[selected])

            return token_id, measured, candidate_ids

        @torch.no_grad()
        def generate(self, prompt, max_new_tokens=64):
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            generated = input_ids

            for _ in range(max_new_tokens):
                result = self.model(
                    input_ids=generated,
                    attention_mask=attention_mask,
                )

                logits = result.logits[:, -1, :]
                logits = logits / self.temperature

                if self.top_k > 0:
                    values, indices = torch.topk(
                        logits, min(self.top_k, logits.shape[-1])
                    )
                    restricted = torch.full_like(logits, float("-inf"))
                    restricted.scatter_(1, indices, values)
                    logits = restricted

                digital_probs = torch.softmax(logits, dim=-1)[0]
                digital_probs_np = digital_probs.float().cpu().numpy()

                if self.optical_mix <= 0.0:
                    # Pure digital sampling
                    next_token = int(
                        self.rng.choice(
                            len(digital_probs_np), p=digital_probs_np
                        )
                    )
                else:
                    if self.use_topk_optical:
                        token_id, measured, candidate_ids = self._optical_topk_sample(
                            digital_probs_np
                        )
                    else:
                        token_id, measured = self.optical_sampler.sample(
                            digital_probs_np, rng=self.rng
                        )
                        candidate_ids = None

                    if self.optical_mix >= 1.0:
                        next_token = token_id
                    else:
                        # Mix digital and optical distributions over the same top-k candidates
                        if candidate_ids is None or measured is None:
                            # Fallback to digital
                            next_token = int(
                                self.rng.choice(
                                    len(digital_probs_np), p=digital_probs_np
                                )
                            )
                        else:
                            digital_topk = digital_probs_np[candidate_ids]
                            digital_topk = np.maximum(digital_topk, 0.0)
                            s = digital_topk.sum()
                            if s > 0:
                                digital_topk /= s

                            mixed = (
                                (1.0 - self.optical_mix) * digital_topk
                                + self.optical_mix * measured
                            )
                            mixed = np.maximum(mixed, 0.0)
                            mixed /= mixed.sum()

                            chosen_local = int(self.rng.choice(len(mixed), p=mixed))
                            next_token = int(candidate_ids[chosen_local])

                next_tensor = torch.tensor(
                    [[next_token]], dtype=torch.long, device=self.device
                )
                generated = torch.cat([generated, next_tensor], dim=1)

                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones(
                                (1, 1), dtype=attention_mask.dtype, device=self.device
                            ),
                        ],
                        dim=1,
                    )

                if next_token == self.tokenizer.eos_token_id:
                    break

            return self.tokenizer.decode(
                generated[0], skip_special_tokens=True
            )

else:
    OpticalGPTGenerator = None  # type: ignore

# ===============================================================
# MAIN
# ===============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "GPT generation with optical-bench probability sampling."
        )
    )

    # MODEL
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Hugging Face model name or path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of optical computing is",
        help="Prompt text.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k for digital logits filtering.",
    )
    parser.add_argument(
        "--optical-mix",
        type=float,
        default=0.1,
        help="Mixing weight for optical distribution (0=digital, 1=optical).",
    )
    parser.add_argument(
        "--optical-topk",
        type=int,
        default=64,
        help="Number of top candidates to load onto optical bench.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed.",
    )

    # CAMERA
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic camera instead of physical camera.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index for real camera.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=480,
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

    # OPTICAL BENCH
    parser.add_argument(
        "--modes-x",
        type=int,
        default=8,
        help="Number of optical modes along x (columns).",
    )
    parser.add_argument(
        "--modes-y",
        type=int,
        default=8,
        help="Number of optical modes along y (rows).",
    )
    parser.add_argument(
        "--temporal-bins",
        type=int,
        default=1,
        help="Number of temporal bins (for streaming large vocab).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=255.0,
        help="Scale factor for probability -> intensity mapping.",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=0.02,
        help="Settle time after ITO pattern update (seconds).",
    )

    # ITO
    parser.add_argument(
        "--serial",
        default=None,
        help="ITO controller COM port (e.g. COM7 or /dev/ttyUSB0).",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200,
    )

    args = parser.parse_args()

    if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "Install torch and transformers:\n"
            "python -m pip install torch transformers"
        )

    # MODEL & TOKENIZER
    print("Loading model:", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # CAMERA
    if args.synthetic:
        print("CAMERA MODE: SYNTHETIC")
        camera = SyntheticCamera(
            width=args.camera_width,
            height=args.camera_height,
            noise=0.02,
            seed=args.seed,
            structured=True,
        )
    else:
        print("CAMERA MODE: REAL")
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
        roi=None,
    )

    # OPTICAL SAMPLER
    sampler = OpticalProbabilitySampler(
        bench=bench,
        ito=ito,
        rows=args.modes_y,
        cols=args.modes_x,
        temporal_bins=args.temporal_bins,
        scale=args.scale,
        settle_time=args.settle,
        background=None,
    )

    # GENERATOR
    generator = OpticalGPTGenerator(
        model=model,
        tokenizer=tokenizer,
        optical_sampler=sampler,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        optical_mix=args.optical_mix,
        seed=args.seed,
        use_topk_optical=True,
        optical_topk=args.optical_topk,
    )

    print()
    print("Prompt:", args.prompt)
    print("Max new tokens:", args.max_new_tokens)
    print("Temperature:", args.temperature)
    print("Top-k:", args.top_k)
    print("Optical mix:", args.optical_mix)
    print("Optical top-k:", args.optical_topk)
    print("Optical bench:", args.modes_y, "x", args.modes_x)
    print("Temporal bins:", args.temporal_bins)
    print()

    try:
        text = generator.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
        print("Generated text:")
        print(text)
    finally:
        camera.close()
        ito.close()

if __name__ == "__main__":
    main()
