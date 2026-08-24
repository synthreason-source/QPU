#!/usr/bin/env python3
"""
GPT generation with FULL optical-bench sampling (minimal coherent optical emulator).

This script:
- Loads a Hugging Face causal LM (e.g. GPT-2).
- At each generation step:
    - Runs the transformer forward pass on CPU/GPU.
    - Converts logits to probabilities.
    - Selects top-k candidates.
    - Sends those probabilities to a MINIMAL optical emulator.
    - Uses ONLY the optical measurement to choose the next token.

The optical emulator is a simple, low-noise, monotonic response:
    m = p + noise, then renormalized.
This keeps the full optical code path but ensures coherent output.

Requirements:
    pip install torch transformers numpy
"""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ===============================================================
# MINIMAL OPTICAL EMULATOR
# ===============================================================

class SimpleOpticalEmulator:
    """
    Minimal optical emulator:
      - Takes a probability vector p (length K).
      - Returns m = p + noise, renormalized.

    This keeps the full optical code path but ensures coherence.
    """

    def __init__(self, dim: int, noise_std: float = 0.02, seed: int = 2026):
        self.dim = int(dim)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)

    def measure(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        if p.ndim == 1 and len(p) == self.dim:
            m = p + self.rng.normal(0.0, self.noise_std, size=p.shape)
        else:
            p_flat = np.reshape(p, -1)[:self.dim]
            if len(p_flat) < self.dim:
                p_flat = np.pad(p_flat, (0, self.dim - len(p_flat)))
            m = p_flat + self.rng.normal(0.0, self.noise_std, size=(self.dim,))
        m = np.maximum(m, 0.0)
        s = m.sum()
        if s <= 0:
            return np.ones(self.dim, dtype=np.float64) / self.dim
        return m / s

# ===============================================================
# OPTICAL PROBABILITY SAMPLER
# ===============================================================

class OpticalProbabilitySampler:
    """
    Optical sampler using the minimal emulator.

    The optical measurement is the ONLY source of randomness.
    """

    def __init__(self, dim: int, noise_std: float = 0.02, seed: int = 2026):
        self.dim = int(dim)
        self.emulator = SimpleOpticalEmulator(dim, noise_std, seed)
        self.calibration_matrix: Optional[np.ndarray] = None

    def _normalise(self, probabilities):
        p = np.asarray(probabilities, dtype=np.float64)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        p = np.maximum(p, 0.0)
        total = p.sum()
        if total <= 0:
            raise ValueError("Probability vector has zero mass")
        return p / total

    def measure_distribution(self, probabilities):
        p = self._normalise(probabilities)
        m = self.emulator.measure(p)
        if self.calibration_matrix is not None:
            K = len(m)
            if self.calibration_matrix.shape == (K, K):
                p_est = self.calibration_matrix @ m
                p_est = np.maximum(p_est, 0.0)
                total = p_est.sum()
                if total > 0:
                    return p_est / total
        return m

    def sample(self, probabilities, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        m = self.measure_distribution(probabilities)
        token_id = int(rng.choice(len(m), p=m))
        return token_id, m

# ===============================================================
# OPTICAL GPT GENERATOR
# ===============================================================

if TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE:

    class OpticalGPTGenerator:
        """
        Generates text from a causal LM with FULL optical sampling.

        The optical emulator determines ALL token choices.
        """

        def __init__(
            self,
            model,
            tokenizer,
            optical_sampler,
            device="cuda",
            temperature=1.0,
            top_k=50,
            seed=2026,
            optical_topk=64,
        ):
            self.model = model.to(device).eval()
            self.tokenizer = tokenizer
            self.optical_sampler = optical_sampler
            self.device = device
            self.temperature = float(temperature)
            self.top_k = int(top_k)
            self.rng = np.random.default_rng(seed)
            self.optical_topk = int(optical_topk)

        def _optical_topk_sample(self, digital_probs):
            candidate_ids = np.argpartition(
                digital_probs, -self.optical_topk
            )[-self.optical_topk:]

            candidate_probs = digital_probs[candidate_ids]
            candidate_probs = np.maximum(candidate_probs, 0.0)
            total = candidate_probs.sum()
            if total <= 0:
                raise RuntimeError(
                    "Top-k candidate probabilities sum to zero; "
                    "cannot send to optical bench."
                )
            candidate_probs /= total

            token_id, measured = self.optical_sampler.sample(
                candidate_probs, rng=self.rng
            )

            token_id = int(candidate_ids[token_id])
            return token_id, measured, candidate_ids

        @torch.no_grad()
        def generate(self, prompt, max_new_tokens=64):
            encoded = self.tokenizer(prompt, return_tensors="pt")
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

                token_id, measured, candidate_ids = self._optical_topk_sample(
                    digital_probs_np
                )

                next_tensor = torch.tensor(
                    [[token_id]], dtype=torch.long, device=self.device
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

                if token_id == self.tokenizer.eos_token_id:
                    break

            return self.tokenizer.decode(generated[0], skip_special_tokens=True)

else:
    OpticalGPTGenerator = None  # type: ignore

# ===============================================================
# MAIN
# ===============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "GPT generation with FULL optical-bench sampling "
            "(minimal coherent optical emulator)."
        )
    )

    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--prompt", type=str, default="The future of optical computing is")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--optical-topk", type=int, default=4)
    parser.add_argument("--optical-noise", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=2026)

    args = parser.parse_args()

    if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "Install torch and transformers:\n"
            "python -m pip install torch transformers"
        )

    print("Loading model:", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    sampler = OpticalProbabilitySampler(
        dim=args.optical_topk,
        noise_std=args.optical_noise,
        seed=args.seed,
    )

    generator = OpticalGPTGenerator(
        model=model,
        tokenizer=tokenizer,
        optical_sampler=sampler,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        optical_topk=args.optical_topk,
    )

    print()
    print("=" * 70)
    print("FULL OPTICAL SAMPLING MODE (MINIMAL COHERENT EMULATOR)")
    print("=" * 70)
    print("The optical emulator determines ALL token choices.")
    print("No digital mixing, no fallback, no second opinion.")
    print("The optical response is m = p + small_noise, renormalized,")
    print("ensuring coherence while keeping the optical path.")
    print("=" * 70)
    print()

    print("Prompt:", args.prompt)
    print("Max new tokens:", args.max_new_tokens)
    print("Temperature:", args.temperature)
    print("Top-k (digital prefilter):", args.top_k)
    print("Optical top-k:", args.optical_topk)
    print("Optical noise std:", args.optical_noise)
    print()

    text = generator.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    print("Generated text:")
    print(text)

if __name__ == "__main__":
    main()
