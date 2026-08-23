import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm

# Import the real/synthetic optical bench pipeline components
from quantum_optical_bench import SyntheticCamera, OpticalBench, ITOController, TemporalOpticalEngine, QiskitOpticalInterface

# --- 1. LIVE OPTICAL BENCH FEATURE SEEDER ---
class LiveOpticalBenchSeeder(nn.Module):
    """
    Directly queries the physical or synthetic optical bench engine 
    while shielding PyTorch's autograd graph from memory leaks using torch.no_grad().
    """
    def __init__(self, modes_x: int = 16, modes_y: int = 16, temporal_bins: int = 2):
        super().__init__()
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.temporal_bins = temporal_bins
        
        # Initialize the optical hardware / synthetic bench stream once
        self.camera = SyntheticCamera(width=320, height=240, noise=0.03, seed=2026, structured=True)
        self.ito = ITOController(port="COM7")
        self.bench = OpticalBench(camera=self.camera, rows=self.modes_y, cols=self.modes_x)

    def forward(self, scalar_input: torch.Tensor) -> torch.Tensor:
        # Prevent autograd from storing execution history of OpenCV/Numpy optical frames
        with torch.no_grad():
            val = scalar_input.detach().cpu().mean().item()
            pattern_index = int(abs(val) * 100) % 16
            
            # Execute the temporal optical engine stream directly
            engine = TemporalOpticalEngine(
                bench=self.bench,
                ito=self.ito,
                spatial_rows=self.modes_y,
                spatial_cols=self.modes_x,
                temporal_bins=self.temporal_bins,
                pattern_type="checker",
                pattern_index=pattern_index,
                prime_only=True
            )
            
            optical_result = engine.process(save_top=8)
            top_primes = optical_result["top_prime_modes"]
            total_power = optical_result["total_power"] + 1e-6
            
            if len(top_primes) >= 2:
                exp_z1 = float(top_primes[0][0]) / total_power
                exp_z12 = float(top_primes[1][0]) / total_power
            else:
                exp_z1, exp_z12 = 0.0, 0.0

        return torch.tensor([exp_z1, exp_z12], dtype=torch.float32, device=scalar_input.device)

    def __del__(self):
        try:
            self.camera.close()
            self.ito.close()
        except Exception:
            pass


# --- 2. OPTICAL BENCH HYBRID MODEL ---
class OpticalBenchTextGenerator(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.quantum_seeder = LiveOpticalBenchSeeder()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim + 2, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, text_tensor: torch.Tensor, hidden=None):
        batch_size, seq_len = text_tensor.shape
        embedded = self.embedding(text_tensor)
        
        scalar_seed = text_tensor[:, 0].float().mean() / 1000.0
        q_features = self.quantum_seeder(scalar_seed)
        
        q_broadcast = q_features.repeat(batch_size, seq_len, 1)
        lstm_input = torch.cat([embedded, q_broadcast], dim=-1)
        
        out, hidden = self.lstm(lstm_input, hidden)
        logits = self.fc(out)
        return logits, hidden


def build_correlation_matrix(words):
    trigrams = defaultdict(Counter)
    bigrams = defaultdict(Counter)
    for i in range(len(words) - 2):
        w1, w2, w3 = words[i], words[i+1], words[i+2]
        trigrams[(w1, w2)][w3] += 1
        bigrams[w1][w2] += 1
    return dict(trigrams), dict(bigrams)


def generate_correlated_text(model, prompt_str, trigrams, bigrams, vocab_to_int, int_to_vocab, max_words=15, temperature=0.7):
    model.eval()
    words = prompt_str.split()
    with torch.no_grad():
        for _ in range(max_words):
            next_word = None
            if len(words) >= 2 and (words[-2], words[-1]) in trigrams:
                context = (words[-2], words[-1])
                possible_next = trigrams[context]
                candidates, counts = zip(*possible_next.items())
                probs = [c / sum(counts) for c in counts]
                next_word = candidates[torch.multinomial(torch.tensor(probs), num_samples=1).item()]

            if not next_word and len(words) >= 1 and words[-1] in bigrams:
                context = words[-1]
                possible_next = bigrams[context]
                candidates, counts = zip(*possible_next.items())
                probs = [c / sum(counts) for c in counts]
                next_word = candidates[torch.multinomial(torch.tensor(probs), num_samples=1).item()]

            if not next_word:
                prompt_indices = [vocab_to_int.get(w, 0) for w in words[-5:]]
                input_tensor = torch.tensor([prompt_indices], dtype=torch.long)
                logits, _ = model(input_tensor)
                probs = torch.softmax(logits[0, -1, :] / temperature, dim=-1)
                next_word_idx = torch.multinomial(probs, num_samples=1).item()
                next_word = int_to_vocab.get(next_word_idx, "the")

            words.append(next_word)
    return " ".join(words)


if __name__ == "__main__":
    try:
        with open("singlekb.txt", "r", encoding="utf-8") as file:
            dataset_corpus = file.read()
    except FileNotFoundError:
        dataset_corpus = "quantum computing and neural networks integration allows hybrid workflow optimization cascades."

    words = dataset_corpus.split()
    vocab = sorted(list(set(words)))
    vocab_size = len(vocab)
    
    vocab_to_int = {word: i for i, word in enumerate(vocab)}
    int_to_vocab = {i: word for i, word in enumerate(vocab)}

    trigrams, bigrams = build_correlation_matrix(words)

    seq_length = 8
    inputs, targets = [], []
    for i in range(len(words) - seq_length):
        inputs.append([vocab_to_int[w] for w in words[i:i + seq_length]])
        targets.append([vocab_to_int[w] for w in words[i + 1:i + seq_length + 1]])

    X = torch.tensor(inputs, dtype=torch.long)
    Y = torch.tensor(targets, dtype=torch.long)

    model = OpticalBenchTextGenerator(vocab_size, embed_dim=16, hidden_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print("--- Training Hybrid Network with Live Optical Bench Streaming & Tqdm ---")
    
    batch_size = 512
    epochs = 5
    
    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        
        # Wrap the batch iterator in a tqdm progress bar
        batch_indices = range(0, len(X), batch_size)
        progress_bar = tqdm(batch_indices, desc=f"Epoch {epoch+1}/{epochs}")
        
        num_batches = 0
        for i in progress_bar:
            xb = X[i:i+batch_size]
            yb = Y[i:i+batch_size]
            
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits.view(-1, vocab_size), yb.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update tqdm progress description with current loss
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_loss = total_loss / max(1, num_batches)
        print(f"Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f}\n")

    while True:
        test_prompt = input("USER: ")
        result = generate_correlated_text(model, test_prompt, trigrams, bigrams, vocab_to_int, int_to_vocab, max_words=500)
        print(f"\nGenerated Output: {result}\n")
