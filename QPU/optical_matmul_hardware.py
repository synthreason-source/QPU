"""
optical_matmul_hardware.py
============================
Runs the 8x8 binary matrix multiply C = A @ B as an actual optical
experiment through the SLM + camera abstraction:

  - B is displayed as a transmittance mask on the SLM (0/1 -> 0.0/1.0).
  - For each row i of A, the illumination source intensities are set
    to A[i, :], and the camera captures the column-summed intensities:
        raw[j] = illumination @ mask[:, j]  (+ noise, + quantization)
  - Raw ADC counts are calibrated back to an integer estimate and
    rounded, then compared to the exact classical result.

This uses MockOpticalBench (real physics + realistic noise), so it
runs immediately with no hardware attached. Swap in the real SLM/
Camera subclasses from optical_hardware_interface.py to run this on
an actual bench - the experiment logic below doesn't change, only
which SLM/Camera objects you construct.
"""

import numpy as np
import matplotlib.pyplot as plt

from optical_hardware_interface import MockOpticalBench, MockSLM, MockCamera

rng = np.random.default_rng(7)
N = 8
A = rng.integers(0, 2, size=(N, N))
B = rng.integers(0, 2, size=(N, N))
C_classical = A @ B


def run_optical_matmul(slm, camera, bench_shape, A, B, n_frames=1):
    """Displays B on the SLM, sweeps each row of A through as
    illumination, and reads back an integer-valued result matrix.
    Averages `n_frames` camera captures per row to reduce noise."""
    n_rows, n_cols = A.shape
    slm.set_pattern(B.astype(float))  # 0/1 bits -> 0.0/1.0 transmittance

    raw = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        camera.bench.set_illumination(A[i, :].astype(float))
        frames = np.stack([camera.capture()[0] for _ in range(n_frames)])
        raw[i, :] = frames.mean(axis=0)
    return raw


def calibrate(bench, camera, n_rows, n_cols, n_frames=8):
    """Measure ADC counts for a known illumination=all-ones,
    mask=all-ones case to find counts-per-unit-signal, so raw counts
    can be converted back to an estimated integer sum."""
    bench.set_pattern(np.ones((n_rows, n_cols)))
    bench.set_illumination(np.ones(n_rows))
    frames = np.stack([camera.capture()[0] for _ in range(n_frames)])
    counts_for_full_sum = frames.mean(axis=0).mean()  # ideal value here is n_rows
    counts_per_unit = counts_for_full_sum / n_rows
    return counts_per_unit


bench = MockOpticalBench(n_rows=N, n_cols=N, gain=40.0, read_noise_std=25.0,
                          shot_noise_scale=3.0, adc_bits=12, seed=42)
slm = MockSLM(bench)
camera = MockCamera(bench)

counts_per_unit = calibrate(bench, camera, N, N)

print("=== Single-shot capture (no averaging) ===")
raw_1 = run_optical_matmul(slm, camera, (N, N), A, B, n_frames=1)
decoded_1 = np.round(raw_1 / counts_per_unit).astype(int)
err_1 = np.abs(C_classical - decoded_1)
print("Max error, 1 frame/row:", err_1.max(), " | exact match:", np.array_equal(C_classical, decoded_1))

print("\n=== Averaged over 64 frames/row (typical real-experiment noise mitigation) ===")
raw_16 = run_optical_matmul(slm, camera, (N, N), A, B, n_frames=64)
decoded_16 = np.round(raw_16 / counts_per_unit).astype(int)
err_16 = np.abs(C_classical - decoded_16)
print("Max error, 64 frames/row:", err_16.max(), " | exact match:", np.array_equal(C_classical, decoded_16))

# -----------------------------------------------------------------
# Plot
# -----------------------------------------------------------------

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
fig.suptitle("Optical Vector-Matrix Multiplier (SLM mask + camera readout, simulated bench)",
             fontsize=14)

def show_bits(ax, M, title, cmap="gray", vmax=1):
    ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax)
    for (r, c), v in np.ndenumerate(M):
        ax.text(c, r, str(v), ha="center", va="center", color="red", fontsize=8)
    ax.set_title(title); ax.set_xlabel("column"); ax.set_ylabel("row")
    ax.invert_yaxis()

show_bits(axes[0, 0], A, "ILLUMINATION PATTERNS\nA (one row used per shot)")
show_bits(axes[0, 1], B, "SLM MASK\nB transmittance (0/1)")
show_bits(axes[0, 2], C_classical, "DIGITAL AFTER\nC = A @ B (exact)", cmap="inferno", vmax=C_classical.max())
axes[0, 3].axis("off")
axes[0, 3].text(0.05, 0.5,
    "Physics used:\n  raw[j] = illumination . mask[:,j]\n"
    "+ shot noise + read noise + ADC\n\n"
    "This is the real Goodman & Athale\n"
    "incoherent optical vector-matrix\n"
    "multiplier architecture - no\n"
    "interference/superposition needed,\n"
    "just intensity summation.",
    fontsize=11, va="center")

show_bits(axes[1, 0], decoded_1, "DECODED\n1 frame/row", cmap="inferno", vmax=C_classical.max())
show_bits(axes[1, 1], decoded_16, "DECODED\n64 frames averaged", cmap="inferno", vmax=C_classical.max())

im = axes[1, 2].imshow(err_1, cmap="coolwarm", vmin=0, vmax=max(1, err_1.max()))
for (r, c), v in np.ndenumerate(err_1):
    axes[1, 2].text(c, r, str(v), ha="center", va="center", color="black", fontsize=8)
axes[1, 2].set_title("ABSOLUTE ERROR\n1 frame/row"); axes[1, 2].invert_yaxis()

axes[1, 3].axis("off")
axes[1, 3].text(0.05, 0.5,
    f"1 frame/row : max err = {err_1.max()}, exact = {np.array_equal(C_classical, decoded_1)}\n"
    f"64 frames/row: max err = {err_16.max()}, exact = {np.array_equal(C_classical, decoded_16)}\n\n"
    "Real cameras/SLMs will have their\n"
    "own gain, dark current, nonlinearity\n"
    "and settling-time quirks -\n"
    "recalibrate against your own bench.",
    fontsize=11, va="center")

plt.tight_layout()
plt.savefig("optical_slm_camera_matmul.png", dpi=150)
print("\nSaved plot to optical_slm_camera_matmul.png")
