"""
Filter-Gated Hologram Transpose
=================================
The filters aren't just upstream preprocessing here -- they DRIVE the
transpose. A site's value only gets relocated to its mirror position
(x,y) -> (y,x) if that source site survived BOTH the 0.1 and 0.2 filters.
Anything either filter filtered contributes nothing to the transposed
output -- it does not move, it is not read, it stays zero.

    filtered_transpose[x,y] = F[y,x]   if filter_map[y,x] == SURVIVED
                             = 0        otherwise

This makes the filters the actual gating mechanism of the transpose
operation, not just a preprocessing step that happens to run first.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

np.set_printoptions(precision=2, suppress=True)

filtered_01, filtered_02, SURVIVED = 0, 1, 2


def build_hole_plate(n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.round(rng.uniform(0, 1, size=(n, n)), 2)


def apply_filter(matrix: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(matrix > threshold, matrix, 0.0)


def build_filter_map(after_1: np.ndarray, after_2: np.ndarray) -> np.ndarray:
    label = np.full(after_1.shape, SURVIVED, dtype=int)
    label[after_1 == 0] = filtered_01
    label[(after_1 != 0) & (after_2 == 0)] = filtered_02
    return label


def build_phase_mask(n: int) -> np.ndarray:
    xs = np.arange(n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    return 0.4 * X + 0.15 * Y ** 2


def build_hologram(amplitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return amplitude * np.exp(1j * phase)


def filtered_transpose(F: np.ndarray, filter_map: np.ndarray) -> np.ndarray:
    """
    THE core operation: filter_map gates which sites get transposed.
    filtered_transpose[x,y] reads F[y,x] ONLY if the source site (y,x)
    survived both filters. Otherwise the destination is just 0 -- the
    filter filtered that data before it ever had a chance to move.
    """
    n = F.shape[0]
    out = np.zeros_like(F)
    for x in range(n):
        for y in range(n):
            if filter_map[y, x] == SURVIVED:
                out[x, y] = F[y, x]
            # else: filter filtered the source -> nothing to transpose, stays 0
    return out


def verify(F: np.ndarray, filter_map: np.ndarray, FT_gated: np.ndarray) -> None:
    n = F.shape[0]
    errors = 0
    for x in range(n):
        for y in range(n):
            source_survived = filter_map[y, x] == SURVIVED
            if source_survived:
                ok = np.isclose(FT_gated[x, y], F[y, x])
            else:
                ok = FT_gated[x, y] == 0
            if not ok:
                errors += 1
    print(f"Gated-transpose verification: {errors} errors across {n*n} sites "
          f"(0 expected -- every moved value came from a filter-surviving site, "
          f"every filter-filtered site produced a 0 with nothing relocated).")
    assert errors == 0


def plot(filter_map: np.ndarray, F: np.ndarray, FT_gated: np.ndarray, out_path: str) -> None:
    n = filter_map.shape[0]
    vmax = np.abs(F).max()
    colors = ListedColormap(["#9aa0a6", "#e07b39", "#2f9e44"])
    labels = {0: "0.1", 1: "0.2", 2: "ok"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    ax0 = axes[0]
    ax0.imshow(filter_map, cmap=colors, vmin=0, vmax=2)
    for x in range(n):
        for y in range(n):
            ax0.text(y, x, labels[filter_map[x, y]], ha="center", va="center",
                      color="white", fontsize=9, fontweight="bold")
    ax0.set_title("Filter map (the gate)\ngray=0.1 kill, orange=0.2 kill, green=survives")
    ax0.set_xlabel("y"); ax0.set_ylabel("x")
    ax0.set_xticks(range(n)); ax0.set_yticks(range(n))

    def annotated(ax, M, title):
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=vmax)
        for x in range(n):
            for y in range(n):
                val = M[x, y]
                c = "white" if val < vmax * 0.6 else "black"
                ax.text(y, x, f"{val:.2f}", ha="center", va="center", color=c, fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("y"); ax.set_ylabel("x")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        return im

    annotated(axes[1], np.abs(F), "|F(x,y)| before\n(ungated hologram)")
    im2 = annotated(axes[2], np.abs(FT_gated), "|filtered_transpose|\nonly filter-survivors moved")
    fig.colorbar(im2, ax=axes[1:], shrink=0.8, label="amplitude")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"\nSaved figure to {out_path}")


def run(n: int = 6, filter_1: float = 0.1, filter_2: float = 0.2,
        out_path: str = "filtered_transpose_result.png") -> None:
    hole_plate = build_hole_plate(n)
    after_1 = apply_filter(hole_plate, filter_1)
    after_2 = apply_filter(after_1, filter_2)
    A = after_2
    filter_map = build_filter_map(after_1, after_2)

    Phi = build_phase_mask(n)
    F = build_hologram(A, Phi)

    FT_gated = filtered_transpose(F, filter_map)

    print("filter_map (0=filtered by 0.1, 1=filtered by 0.2, 2=survived both):")
    print(filter_map)
    print("\n|F| (pre-transpose hologram amplitude):")
    print(np.abs(F))
    print("\n|filtered_transpose| (only filter-surviving sites get relocated):")
    print(np.abs(FT_gated))

    verify(F, filter_map, FT_gated)
    plot(filter_map, F, FT_gated, out_path)


if __name__ == "__main__":
    run()