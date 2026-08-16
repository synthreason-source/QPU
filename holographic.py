
"""
Interference-Based Hologram Transpose
========================================
Unlike the geometric transpose (F.T, or a Dove prism), this version relocates
each voxel to its transposed coordinate purely through wave interference:

  1. Each surviving filtered voxel a_i at (x_i, y_i) becomes a coherent point
     source, but it is placed -- by holographic construction -- at its
     TRANSPOSED target coordinate (y_i, x_i) on a virtual object plane.
  2. A hologram (kinoform, i.e. a complex-transmittance mask -- physically
     the "light valve" array) is synthesized by backward-propagating each of
     those target point sources to a hologram plane and summing their
     spherical wavelets (Huygens-Fresnel principle). This sum IS an
     interference pattern.
  3. The hologram is then "illuminated" and forward-propagated (again via
     Huygens summation of spherical wavelets from every hologram sample) to
     an output plane.
  4. By the reversibility of the holographic recording/reconstruction
     process, the interference of light from every hologram point
     reconstructs bright spots exactly at the transposed coordinates -- with
     brightness proportional to the original filtered amplitude a_i.

Voxels killed by the 0.1 / 0.2 filters have a_i = 0, so they contribute NO
wavelet to the hologram at all -- the filters directly gate which
interference terms exist, and therefore which transposed spots reconstruct.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

LAMBDA = 1.0
K = 2 * np.pi / LAMBDA


def build_hole_plate(n, seed=7):
    rng = np.random.default_rng(seed)
    return np.round(rng.uniform(0, 1, size=(n, n)), 2)


def apply_filter(matrix, threshold):
    return np.where(matrix > threshold, matrix, 0.0)


def huygens_sum(src_x, src_y, src_amp, obs_x, obs_y, z):
    """
    Vectorized Huygens-Fresnel superposition: field at every observation
    point (obs_x, obs_y) produced by coherent spherical wavelets from every
    source point (src_x, src_y) with complex amplitude src_amp, separated by
    propagation distance z. This is literal wave interference -- every
    observation point sums contributions (with phase = k*r) from every
    source point.
    """
    dx = obs_x[:, None] - src_x[None, :]
    dy = obs_y[:, None] - src_y[None, :]
    r = np.sqrt(dx**2 + dy**2 + z**2)
    return (src_amp[None, :] * np.exp(1j * K * r) / r).sum(axis=1)


def run(n=6, filter_1=0.1, filter_2=0.2, pitch=8.0, d=150.0,
        holo_extent=(-20, 60, 2.0), out_extent=(0, 40, 1.0),
        out_path="interference_transpose_result.png"):

    hole_plate = build_hole_plate(n)
    after_1 = apply_filter(hole_plate, filter_1)
    after_2 = apply_filter(after_1, filter_2)
    A = after_2

    xs = np.arange(n)
    Xi, Yi = np.meshgrid(xs, xs, indexing="ij")
    src_x = Xi.flatten() * pitch          # original x-index -> physical x
    src_y = Yi.flatten() * pitch          # original y-index -> physical y
    amp = A.flatten()

    # Transposed TARGET coordinates: swap x<->y for every voxel
    tgt_x = src_y.copy()
    tgt_y = src_x.copy()

    active = amp != 0
    print(f"{active.sum()} / {n*n} voxels survived both filters "
          f"and contribute a wavelet to the hologram.")

    # --- Hologram synthesis plane (samples u,v) ---
    lo, hi, step = holo_extent
    u = np.arange(lo, hi, step)
    U, V = np.meshgrid(u, u, indexing="ij")
    Uf, Vf = U.flatten(), V.flatten()

    print("Synthesizing hologram via Huygens backward-propagation "
          f"from {active.sum()} transposed target sources "
          f"to {len(Uf)} hologram samples...")
    H = huygens_sum(tgt_x[active], tgt_y[active], amp[active].astype(complex),
                     Uf, Vf, d)

    # --- Reconstruction: propagate hologram forward to output plane ---
    lo2, hi2, step2 = out_extent
    o = np.arange(lo2, hi2 + step2, step2)
    Xo, Yo = np.meshgrid(o, o, indexing="ij")
    Xof, Yof = Xo.flatten(), Yo.flatten()

    print(f"Propagating hologram forward to {len(Xof)} output-plane points...")
    E = huygens_sum(Uf, Vf, H, Xof, Yof, d)
    I = np.abs(E).reshape(Xo.shape) ** 2

    # --- Verify: sample the reconstructed field exactly at each transposed target ---
    recon_at_target = np.abs(huygens_sum(Uf, Vf, H, tgt_x, tgt_y, d))
    expected = amp
    mask = active
    corr = np.corrcoef(recon_at_target[mask], expected[mask])[0, 1]
    print(f"\nCorrelation between reconstructed amplitude at each transposed "
          f"target and the original filtered voxel amplitude: {corr:.4f}")
    print("(1.0 = perfect reconstruction; real holograms show <1.0 due to "
          "finite-aperture diffraction sidelobes/crosstalk between sources)")

    print("\nPer-voxel check (source amplitude vs reconstructed amplitude at its transposed target):")
    for i in np.where(mask)[0][:10]:
        print(f"  a_i={expected[i]:.2f} at src=({src_x[i]:.0f},{src_y[i]:.0f}) "
              f"-> target=({tgt_x[i]:.0f},{tgt_y[i]:.0f})  reconstructed |E|={recon_at_target[i]:.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    ax = axes[0]
    im = ax.imshow(A, cmap="viridis", origin="lower")
    ax.set_title("Filtered voxel amplitudes A(x,y)\n(sources feeding the hologram)")
    ax.set_xlabel("y"); ax.set_ylabel("x")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax2 = axes[1]
    im2 = ax2.imshow(I.T, origin="lower", cmap="inferno",
                      extent=[lo2, hi2, lo2, hi2])
    ax2.scatter(tgt_x[mask], tgt_y[mask], s=25, facecolors="none",
                edgecolors="cyan", linewidths=1.0, label="expected transposed positions")
    ax2.set_title("Reconstructed intensity at output plane\n(interference of the hologram)")
    ax2.set_xlabel("physical x"); ax2.set_ylabel("physical y")
    ax2.legend(loc="upper right", fontsize=8)
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    run()