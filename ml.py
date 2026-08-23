#!/usr/env python3
"""
Text generation via optical quantum bench and trigram token mapping.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import numpy as np

from quantum_optical_bench import (
    GeneralQuantumComputer,
    OpticalBackend,
    OpticalBench,
    OpticalCamera,
    SyntheticCamera,
    ITOController,
    parse_q_file,
    parse_operations,
)


def load_trigram_vocab(file_path: str | Path):
    """Loads a text file and builds trigram word sequences as tokens."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    
    text = path.read_text(encoding="utf-8")
    words = text.split()
    
    # Build word-level trigrams
    trigrams = []
    for i in range(len(words) - 2):
        trigrams.append((words[i], words[i+1], words[i+2]))
        
    if not trigrams:
        # Fallback if corpus is too short
        trigrams = [tuple(words + [''] * (3 - len(words)))]
        
    unique_tokens = list(set(trigrams))
    return words, unique_tokens


def generate_text_from_counts(counts: dict[str, int], vocab: list, length: int = 15) -> str:
    """Samples from optical measurement counts to generate a sequence of trigram words."""
    # Convert counts to probability distribution
    bitstrings = list(counts.keys())
    probabilities = np.array(list(counts.values()), dtype=np.float64)
    probabilities /= probabilities.sum()
    
    generated_words = []
    # Start with a random trigram from vocab
    current_trigram = random.choice(vocab)
    generated_words.extend(current_trigram)
    
    for _ in range(length):
        # Sample a bitstring index based on quantum measurement counts
        chosen_bitstring = np.random.choice(bitstrings, p=probabilities)
        token_index = int(chosen_bitstring, 2) % len(vocab)
        
        next_trigram = vocab[token_index]
        # Append the final word of the selected trigram to maintain flow
        generated_words.append(next_trigram[-1])
        
    return " ".join(generated_words)


def main():
    parser = argparse.ArgumentParser(description="Generate text using optical quantum counts.")
    parser.add_argument("--corpus", type=str, required=True, help="Path to text dataset file.")
    parser.add_argument("--qubits", type=int, default=3, help="Number of qubits.")
    parser.add_argument("--shots", type=int, default=1024, help="Measurement shots.")
    parser.add_argument("--script", type=str, default=None, help="Optional .q script for circuit preparation.")
    parser.add_argument("--length", type=int, default=20, help="Number of generated words.")
    args = parser.parse_args()

    # 1. Load dataset and create vocabulary
    words, vocab = load_trigram_vocab(args.corpus)
    print(f"Loaded corpus with {len(words)} words. Unique trigram tokens: {len(vocab)}")

    # 2. Setup simulated optical backend / computer
    camera = SyntheticCamera(width=256, height=256, seed=2026)
    ito = ITOController(port="COM7")
    bench = OpticalBench(camera=camera, rows=16, cols=16)
    
    optical_backend = OpticalBackend(
        bench=bench,
        ito=ito,
        num_qubits=args.qubits,
        spatial_rows=16,
        spatial_cols=16,
        temporal_bins=2,
    )

    computer = GeneralQuantumComputer(
        num_qubits=args.qubits,
        optical_backend=optical_backend,
    )

    # 3. Build circuit from script or default
    if args.script:
        ops = parse_q_file(args.script)
        qc = parse_operations(ops, args.qubits)
    else:
        qc = computer.build_circuit()

    # 4. Execute via optical backend
    result = computer.run_optical(qc, shots=args.shots)
    camera.close()
    ito.close()

    # 5. Generate text from measurement counts
    output_text = generate_text_from_counts(result.counts, vocab, length=500)
    
    print("\n" + "=" * 80)
    print("GENERATED TEXT SEQUENCE:")
    print("=" * 80)
    print(output_text)


if __name__ == "__main__":
    main()