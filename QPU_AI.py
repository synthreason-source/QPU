import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

# ----------------------------------------------------------------------
# 1. UNCHANGED from the original file. These stay the only functions
#    that ever touch the matrix multiply that produces a next-token
#    distribution. Everything added below only decides which row of B
#    gets handed to optical_mac_reference -- it never picks a token
#    index directly.
# ----------------------------------------------------------------------

def encode_same_plane(values, filters):
    if values.shape != filters.shape:
        raise ValueError("values and filters must have identical shapes")
    if np.any(filters < 0.0) or np.any(filters > 1.0):
        raise ValueError("filters must be within [0, 1]")
    return values * filters


def optical_mac_reference(A, B):
    """Shared-index optical MAC: sum_j A[:, j] (x) B[j, :].
    This is the ONLY place a token index is derived from in this file."""
    m, p = A.shape
    p2, n = B.shape
    if p != p2:
        raise ValueError("inner dimensions must match")
    channels = np.empty((p, m, n), dtype=float)
    for j in range(p):
        channels[j] = A[:, j:j + 1] * B[j:j + 1, :]
    return channels.sum(axis=0), channels


# ----------------------------------------------------------------------
# 2. Corpus + trigram statistics (unchanged logic, still plain digital
#    bookkeeping -- no optics yet).
# ----------------------------------------------------------------------

def tokenize(text):
    return text.lower().split()


def build_trigram_tables(tokens):
    vocab = sorted(set(tokens))
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    contexts = []
    context_to_idx = {}
    counts = {}  # context_idx -> {word_idx: count}

    for i in range(len(tokens) - 2):
        ctx = (tokens[i], tokens[i + 1])
        nxt = tokens[i + 2]
        if ctx not in context_to_idx:
            context_to_idx[ctx] = len(contexts)
            contexts.append(ctx)
        cidx = context_to_idx[ctx]
        widx = word_to_idx[nxt]
        counts.setdefault(cidx, {})
        counts[cidx][widx] = counts[cidx].get(widx, 0) + 1

    num_contexts = len(contexts)
    vocab_size = len(vocab)

    B = np.zeros((num_contexts, vocab_size), dtype=float)
    for cidx, wc in counts.items():
        total = sum(wc.values())
        for widx, c in wc.items():
            B[cidx, widx] = c / total

    return vocab, word_to_idx, contexts, context_to_idx, B


# ----------------------------------------------------------------------
# 3. NEW: semantic linked list over contexts.
#
#    Each context (w1, w2) gets a bag-of-words vector built from every
#    trigram window that contains it -- essentially "what tends to
#    co-occur near this context in the corpus". Contexts are then
#    chained into a linked list via greedy nearest-neighbor walk over
#    cosine similarity: node.next always points at the most similar
#    not-yet-visited context. This gives you two things:
#      - a literal linked structure (traverse contexts in semantic
#        order, not corpus order)
#      - a transitive fallback path: if a query context is unseen, hop
#        along the chain from its closest matching neighbor instead of
#        just dead-ending.
# ----------------------------------------------------------------------

class ContextNode:
    __slots__ = ("context", "idx", "vector", "next", "prev")

    def __init__(self, context, idx, vector):
        self.context = context
        self.idx = idx
        self.vector = vector
        self.next = None
        self.prev = None


def build_context_vectors(tokens, contexts, context_to_idx, vocab_size, word_to_idx, window=3):
    """Bag-of-words co-occurrence vector per context: counts of which
    vocab words appear within `window` tokens of each occurrence of
    that context in the corpus. Cheap stand-in for an embedding model
    since no external embedding service is reachable from this
    sandbox."""
    vectors = np.zeros((len(contexts), vocab_size), dtype=float)
    for i in range(len(tokens) - 2):
        ctx = (tokens[i], tokens[i + 1])
        cidx = context_to_idx.get(ctx)
        if cidx is None:
            continue
        lo = max(0, i - window)
        hi = min(len(tokens), i + 2 + window)
        for t in tokens[lo:hi]:
            vectors[cidx, word_to_idx[t]] += 1.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def build_semantic_linked_list(contexts, vectors):
    """Greedy nearest-neighbor chain: start at context 0, always hop
    to the closest not-yet-visited context by cosine similarity."""
    n = len(contexts)
    nodes = [ContextNode(contexts[i], i, vectors[i]) for i in range(n)]
    if n == 0:
        return nodes, None

    visited = np.zeros(n, dtype=bool)
    order = [0]
    visited[0] = True
    current = 0

    for _ in range(n - 1):
        sims = vectors @ vectors[current]
        sims[visited] = -np.inf
        nxt = int(np.argmax(sims))
        nodes[current].next = nodes[nxt]
        nodes[nxt].prev = nodes[current]
        visited[nxt] = True
        order.append(nxt)
        current = nxt

    head = nodes[order[0]]
    return nodes, head


def nearest_known_context(query_vector, vectors, k=1):
    """Direct nearest-neighbor lookup (used as the entry point into the
    linked list when a query context has no exact match)."""
    sims = vectors @ query_vector
    return int(np.argmax(sims)), float(np.max(sims))


def vectorize_query_context(context, word_to_idx, vocab_size):
    v = np.zeros(vocab_size, dtype=float)
    for w in context:
        if w in word_to_idx:
            v[word_to_idx[w]] += 1.0
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


# ----------------------------------------------------------------------
# 4. Bridge: context -> row of B -> optical MAC -> sampled token index.
#    Falls back through the semantic linked list when the exact
#    context was never seen in training.
# ----------------------------------------------------------------------

def context_to_onehot(context, context_to_idx, num_contexts):
    A = np.zeros((1, num_contexts), dtype=float)
    if context in context_to_idx:
        A[0, context_to_idx[context]] = 1.0
        return A, True
    return A, False


def resolve_context_row(context, context_to_idx, vectors, word_to_idx, vocab_size, min_sim=0.05):
    """Return (row_idx, matched_exactly) for the context to feed into B.
    Exact match wins. Otherwise walk to the nearest context in the
    semantic linked list (transitive fallback) if similarity clears a
    floor; below the floor we give up (returns None, False)."""
    num_contexts = len(vectors)
    if context in context_to_idx:
        return context_to_idx[context], True

    qv = vectorize_query_context(context, word_to_idx, vocab_size)
    if np.linalg.norm(qv) == 0 or num_contexts == 0:
        return None, False

    nn_idx, sim = nearest_known_context(qv, vectors)
    if sim < min_sim:
        return None, False
    return nn_idx, False


def next_token_via_optical_mac(context, context_to_idx, vectors, word_to_idx, vocab_size, B, rng):
    num_contexts = B.shape[0]
    row_idx, exact = resolve_context_row(context, context_to_idx, vectors, word_to_idx, vocab_size)
    if row_idx is None:
        return None, None, None, exact

    A = np.zeros((1, num_contexts), dtype=float)
    A[0, row_idx] = 1.0

    filters_A = np.ones_like(A)
    filters_B = np.ones_like(B)
    A_enc = encode_same_plane(A, filters_A)
    B_enc = encode_same_plane(B, filters_B)

    optical_C, channels = optical_mac_reference(A_enc, B_enc)

    dist = optical_C[0]
    total = dist.sum()
    if total <= 0:
        return None, optical_C, channels, exact

    probs = dist / total
    token_idx = int(rng.choice(len(probs), p=probs))
    return token_idx, optical_C, channels, exact


# ----------------------------------------------------------------------
# 5. Generation loop, now reporting how often it had to lean on the
#    semantic fallback vs. an exact trigram hit.
# ----------------------------------------------------------------------

def generate(seed_context, n_tokens, vocab, context_to_idx, vectors, word_to_idx, vocab_size, B, seed=0):
    rng = np.random.default_rng(seed)
    tokens_out = list(seed_context)
    context = seed_context
    exact_hits = 0
    fallback_hits = 0

    for _ in range(n_tokens):
        token_idx, optical_C, channels, exact = next_token_via_optical_mac(
            context, context_to_idx, vectors, word_to_idx, vocab_size, B, rng
        )
        if token_idx is None:
            break
        exact_hits += int(exact)
        fallback_hits += int(not exact)
        word = vocab[token_idx]
        tokens_out.append(word)
        context = (context[1], word)

    return tokens_out, exact_hits, fallback_hits


def score_generation(tokens_out, seed_len):
    """Simple, dependency-free scoring for the seed bruteforce step:
    reward lexical diversity and length, since a stalled/degenerate
    run (repeats or early stop) scores low."""
    body = tokens_out[seed_len:]
    if not body:
        return 0.0
    unique_ratio = len(set(body)) / len(body)
    length_term = len(body)
    return unique_ratio * length_term


def seed_bruteforce(seed_context, n_tokens, vocab, context_to_idx, vectors, word_to_idx, vocab_size, B,
                     seed_range=range(50)):
    """Try many seeds, score each generation, keep the best. Returns
    the winning seed, its generated tokens, and the full score curve
    so you can see how much seed choice actually mattered."""
    best_seed, best_score, best_tokens = None, -1.0, None
    scores = []

    for s in seed_range:
        tokens_out, exact_hits, fallback_hits = generate(
            seed_context, n_tokens, vocab, context_to_idx, vectors, word_to_idx, vocab_size, B, seed=s
        )
        sc = score_generation(tokens_out, len(seed_context))
        scores.append(sc)
        if sc > best_score:
            best_seed, best_score, best_tokens = s, sc, tokens_out

    return best_seed, best_score, best_tokens, scores


def run():
    path = input("Dataset filename: ")
    with open(path, "r", encoding="utf-8") as f:
        corpus = f.read()

    tokens = tokenize(corpus)
    vocab, word_to_idx, contexts, context_to_idx, B = build_trigram_tables(tokens)
    vocab_size = len(vocab)

    print(f"Vocab size: {vocab_size}")
    print(f"Distinct trigram contexts: {len(contexts)}")
    print(f"Transition matrix B shape (contexts x vocab): {B.shape}")

    vectors = build_context_vectors(tokens, contexts, context_to_idx, vocab_size, word_to_idx)
    nodes, head = build_semantic_linked_list(contexts, vectors)
    print(f"Built semantic linked list over {len(nodes)} contexts.")
    if head is not None:
        walk = []
        n = head
        for _ in range(min(5, len(nodes))):
            walk.append(n.context)
            n = n.next
        print("First few hops of the semantic chain:", walk)

    seed_context = ("is", "the")
    best_seed, best_score, best_tokens, scores = seed_bruteforce(
        seed_context, n_tokens=200, vocab=vocab, context_to_idx=context_to_idx,
        vectors=vectors, word_to_idx=word_to_idx, vocab_size=vocab_size, B=B,
        seed_range=range(50),
    )

    print(f"\nBest seed: {best_seed}  (score={best_score:.3f})")
    print("Seed context:", seed_context)
    print("Generated (token indices derived ONLY from optical_mac_reference output):")
    print(" ".join(best_tokens))

    plt.figure(figsize=(8, 4))
    plt.plot(list(range(len(scores))), scores, marker="o", markersize=3)
    plt.axvline(best_seed, color="red", linestyle="--", label=f"best seed = {best_seed}")
    plt.xlabel("seed")
    plt.ylabel("diversity x length score")
    plt.title("Seed bruteforce results")
    plt.legend()
    plt.tight_layout()
    plt.savefig("seed_scores.png", dpi=150)

    return {
        "vocab": vocab,
        "contexts": contexts,
        "B": B,
        "vectors": vectors,
        "generated": best_tokens,
        "best_seed": best_seed,
        "seed_scores": scores,
    }


if __name__ == "__main__":
    run()
