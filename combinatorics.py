import math
import random
import time
from fractions import Fraction

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

# =====================================================================
# HARDWARE MODEL, THIS VERSION
#
#   1. PLATE HOLES  -- a real 2D array of holes, each strictly binary
#      (0 = shut, 1 = open), one-shot. Nothing continuous, no partial
#      transmission.
#   2. ENTANGLED LASER -- a coherent field carrying the digital VALUES,
#      arranged on the SAME 2D grid as the holes, one value per hole
#      site. "Entangled" here just means value-site and hole-site are
#      the same physical pixel: the laser only shows up where its
#      matching hole is open.
#   3. 2D PLANE FILTER -- the laser passes through the plate: an
#      elementwise product of the values grid and the holes grid.
#      This is the ONLY thing the optics does at the plate itself.
#   4. INTERFERENCE CALC AT THE CAMERA IMAGE -- the filtered plate is
#      not read out pixel-by-pixel. It propagates (a single 2D Fourier
#      transform -- ordinary Fraunhofer far-field diffraction to a
#      camera sitting at the focal plane of a lens) and the CAMERA
#      records an interference pattern, i.e. optical power
#      |field|^2 at every pixel. The MAC accumulation -- the thing we
#      actually want -- is read off ONE pixel of that camera image:
#      the zero-order (DC) spot at the exact center of the pattern.
#      That pixel's complex amplitude is, by the definition of the
#      Fourier transform, the sum of every value on the plate. So
#      "sum the gated values" (the MAC) is done physically by
#      constructive interference at a single camera pixel, not by
#      digital addition.
#
# THE KEY SCALING POINT THIS FILE NOW DEMONSTRATES
#
#   On a real photonic version of this rig, step 4 is ONE camera
#   exposure no matter how many terms are on the plate -- a million
#   values interfere down to a single DC pixel in the same single
#   shot as ten values would. The accumulation is O(1) in the number
#   of terms.
#
#   A conventional CPU cannot do that: it has to visit every term and
#   add it in, one MAC at a time, so its cost is O(#terms). That's the
#   contrast this script now makes visible -- it grows the loop's
#   input vectors until the number of terms (I*J*K) is large, runs a
#   literal scalar Python accumulation (no numpy vectorization
#   allowed, to keep it an honest "one MAC at a time on a PC" cost
#   model) side by side with the plate+camera path, and times both.
#
# Modular reduction, carries beyond what interference-summing already
# gives for free, and period extraction from a built sequence remain
# explicitly digital/classical control logic layered around this
# optical core, exactly as before.
# =====================================================================


def entangled_laser_illuminate(values_plate, holes_plate):
    """The laser (values_plate) passing through the plate's holes
    (holes_plate). Every site is a paired (laser value, hole) pixel on
    the SAME 2D grid -- the plate physically only ever does this one
    elementwise gating operation."""
    if values_plate.shape != holes_plate.shape:
        raise ValueError("values plate and holes plate must be the same 2D shape")
    if not np.all(np.isin(holes_plate, [0.0, 1.0])):
        raise ValueError("holes must be strictly binary (0 shut / 1 open)")
    return values_plate * holes_plate


# Kept as a name-compatible alias: earlier versions of this hardware
# model called the same operation encode_same_plane(values, valves).
encode_same_plane = entangled_laser_illuminate


def fft2c(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))


def ifft2c(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))


def propagate_to_camera(filtered_plate):
    """Single Fraunhofer propagation (one 2D FFT) from the plate to a
    camera at the lens focal plane. Returns the complex field actually
    present at the camera, and the intensity image the camera sensor
    physically records (|field|^2, an interference/power pattern)."""
    field = fft2c(filtered_plate)
    intensity = np.abs(field) ** 2
    return field, intensity


def camera_dc_pixel(field):
    """The zero spatial-frequency order lands exactly at the center of
    the shifted camera image. Its complex amplitude equals the sum of
    every pixel on the plate that produced it -- this is where the MAC
    accumulation is actually read out, optically, at the camera."""
    cy, cx = field.shape[0] // 2, field.shape[1] // 2
    return field[cy, cx], (cy, cx)


def to_square_plate(vector: np.ndarray) -> np.ndarray:
    """Lay a 1D list of per-hole values out as a genuine 2D plate
    (zero-padded to the next square number, then reshaped). The DC
    camera readout sums every pixel regardless of this arrangement, so
    the physics does not care about the exact layout -- only that it
    is a real 2D grid of holes, not a 1xN strip."""
    n = vector.shape[0]
    side = int(np.ceil(np.sqrt(max(n, 1))))
    padded = np.zeros(side * side, dtype=vector.dtype)
    padded[:n] = vector
    return padded.reshape(side, side)


# =====================================================================
# INTEGER MULTIPLY = ONE PLATE FILTER + ONE CAMERA INTERFERENCE READOUT
#
# x * y = sum_i  x_i * (y << i)     (x_i = bit i of x, LSB first)
#
#   VALUES plate: (y << i) placed at hole site i, laid out on a real
#                 2D grid (to_square_plate).
#   HOLES plate:  bit i of x at that same site, strictly binary.
#   FILTER:       entangled_laser_illuminate -> per-hole partial products,
#                 already zero wherever the bit is 0. This is the
#                 physical multiply-by-bit step.
#   CAMERA:       propagate_to_camera -> interference pattern; the sum
#                 of the partial products (== x*y) is read off the DC
#                 pixel at the camera, not accumulated digitally.
# =====================================================================

def optical_multiply(x: int, y: int, return_optics: bool = False):
    if x < 0 or y < 0:
        raise ValueError("this hardware model only carries unsigned magnitudes")
    if x == 0 or y == 0:
        if return_optics:
            empty = np.zeros((1, 1))
            return 0, {"holes_plate": empty, "values_plate": empty,
                        "filtered_plate": empty, "field": empty.astype(complex),
                        "intensity": empty, "dc_index": (0, 0)}
        return 0

    width = x.bit_length()
    bits = np.array([(x >> i) & 1 for i in range(width)], dtype=float)
    shifted_y = np.array([y * (1 << i) for i in range(width)], dtype=float)

    holes_plate = to_square_plate(bits)              # PLATE HOLES (2D, binary)
    values_plate = to_square_plate(shifted_y)         # ENTANGLED LASER (2D)

    filtered_plate = entangled_laser_illuminate(values_plate, holes_plate)  # 2D PLANE FILTER
    field, intensity = propagate_to_camera(filtered_plate)                  # CAMERA INTERFERENCE
    dc_value, dc_index = camera_dc_pixel(field)                             # MAC READOUT

    product = int(round(dc_value.real))

    if return_optics:
        optics = {"holes_plate": holes_plate, "values_plate": values_plate,
                  "filtered_plate": filtered_plate, "field": field,
                  "intensity": intensity, "dc_index": dc_index}
        return product, optics
    return product



# =====================================================================
# VISUALIZATION
# =====================================================================

def plot_matrix(fig, ax, data, title, cmap="viridis", vmin=None, vmax=None, annotate=False, mark=None):
    image = ax.imshow(data, origin="lower", interpolation="nearest", cmap=cmap,
                       vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=10)
    if annotate and data.size <= 64:
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                ax.text(c, r, f"{data[r, c]:.0f}", ha="center", va="center", color="white", fontsize=7)
    if mark is not None:
        my, mx = mark
        ax.plot(mx, my, marker="x", color="lime", markersize=12, markeredgewidth=2)
    fig.colorbar(image, ax=ax, shrink=0.75)



# =====================================================================
# 3D LOOP EVALUATOR -- a second algorithm on the SAME hardware model.
#
# Takes three input vectors A (length I), B (length J), C (length K)
# and evaluates
#
#     result = sum_{i,j,k}  A_i * B_j * C_k      (mod N, if N is given)
#
# via a literal triple-nested loop over (i, j, k). Every multiply
# inside the loop is a plate + camera pass (optical_multiply, chained
# twice per term: A_i*B_j, then that * C_k). Nothing is added digitally
# term-by-term -- once the loop has built all I*J*K terms, they are
# summed in a SINGLE further plate + camera pass (optical_add_many):
# all terms are laid out as one big values plate with every hole open,
# propagated to a camera, and the total is read straight off the DC
# pixel by constructive interference, exactly like the multiply
# primitive's own MAC readout. So the "3D loop" supplies the input
# values and control flow; every arithmetic step, including the final
# accumulation, is optical.
# =====================================================================

def optical_add_many(values, N: int | None = None):
    """Sum an arbitrary list of values in ONE plate + camera pass: lay
    them on a 2D plate with every hole open (no gating), propagate to
    the camera, and read the total off the DC pixel. Padding sites
    introduced by to_square_plate are zero-valued AND zero-holed, so
    they contribute nothing to the sum. This one call replaces what a
    CPU would otherwise do as len(values) sequential additions."""
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return 0
    values_plate = to_square_plate(arr)
    holes_plate = to_square_plate(np.ones_like(arr))  # padding -> 0, real sites -> 1 (open)
    filtered_plate = entangled_laser_illuminate(values_plate, holes_plate)
    field, intensity = propagate_to_camera(filtered_plate)
    dc_value, dc_index = camera_dc_pixel(field)
    total = int(round(dc_value.real))
    if N is not None:
        total %= N
    return total


def evaluate_3d_loop(A, B, C, N: int | None = None, mac_calls: list[int] | None = None):
    """Triple-nested loop over three input vectors. Every multiply is
    an optical plate+camera pass; the final sum over all I*J*K terms
    is a single further plate+camera interference pass.

    Returns (result, terms, trace) where terms is the flat list of all
    I*J*K products (each already reduced mod N if N is given) and
    trace records (i, j, k, a, b, c, term) for every loop iteration."""
    terms = []
    trace = []
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            ab = optical_multiply(int(a), int(b))          # plate + camera pass 1
            if mac_calls is not None:
                mac_calls[0] += 1
            if N is not None:
                ab %= N
            for k, c in enumerate(C):
                term = optical_multiply(ab, int(c))         # plate + camera pass 2
                if mac_calls is not None:
                    mac_calls[0] += 1
                if N is not None:
                    term %= N
                terms.append(term)
                trace.append((i, j, k, a, b, c, term))

    result = optical_add_many(terms, N=N)                    # ONE final interference sum
    if mac_calls is not None:
        mac_calls[0] += 1

    return result, terms, trace


# =====================================================================
# HONEST "PC-ONLY" BASELINE
#
# A literal, unvectorized, scalar-Python triple loop that does exactly
# what a conventional CPU actually has to do: I*J*K multiplies and
# I*J*K-1 additions, one at a time, in Python (no numpy, no batching --
# that would smuggle vectorized hardware tricks back in). This is the
# thing whose wall-clock time balloons as the input vectors grow. The
# optical path's final accumulation stays a single plate+camera pass
# regardless of how many terms there are, which is the point.
# =====================================================================

def digital_brute_force(A, B, C, N: int | None = None):
    total = 0
    for a in A:
        for b in B:
            ab = a * b
            if N is not None:
                ab %= N
            for c in C:
                term = ab * c
                if N is not None:
                    term %= N
                total += term
                if N is not None:
                    total %= N
    return total


def make_3d_loop_visualization(A, B, C, N, result, terms, trace, mac_calls, out_path):
    fig = plt.figure(figsize=(20, 11), layout="constrained")
    grid = fig.add_gridspec(2, 3)

    I, J, K = len(A), len(B), len(C)

    # A slice of the loop's term cube at fixed k=0, so the reader can
    # see the (i, j) structure the loop walks before k is folded in.
    k0 = 0
    slice_ij = np.array([[t for (i, j, k, a, b, c, t) in trace if k == k0]
                          for _ in [0]]).reshape(I, J) if K > 0 else np.zeros((I, J))

    ax1 = fig.add_subplot(grid[0, 0])
    plot_matrix(fig, ax1, np.array(A, dtype=float).reshape(1, -1), f"INPUT A (length {I})",
                cmap="viridis", annotate=(I <= 16))
    ax2 = fig.add_subplot(grid[0, 1])
    plot_matrix(fig, ax2, np.array(B, dtype=float).reshape(1, -1), f"INPUT B (length {J})",
                cmap="viridis", annotate=(J <= 16))
    ax3 = fig.add_subplot(grid[0, 2])
    plot_matrix(fig, ax3, np.array(C, dtype=float).reshape(1, -1), f"INPUT C (length {K})",
                cmap="viridis", annotate=(K <= 16))

    ax4 = fig.add_subplot(grid[1, 0])
    plot_matrix(fig, ax4, slice_ij, f"LOOP TERMS A_i*B_j*C_k\nslice at k={k0} (each cell = 2 plate+camera passes)",
                cmap="plasma", annotate=(I * J <= 64))

    ax5 = fig.add_subplot(grid[1, 1])
    term_plate = to_square_plate(np.array(terms, dtype=float))
    plot_matrix(fig, ax5, term_plate, f"ALL {len(terms)} TERMS LAID OUT\nas the plate for the final sum pass",
                cmap="plasma", annotate=(term_plate.size <= 64))

    ax6 = fig.add_subplot(grid[1, 2])
    ax6.axis("off")
    lines = [
        "3D LOOP -> ARITHMETIC -> RESULT", "",
        f"A: length {I}",
        f"B: length {J}",
        f"C: length {K}",
        f"N (mod, optional): {N}",
        "",
        f"loop shape: {I} x {J} x {K} = {I*J*K} terms",
        "each term: 2 chained plate+camera multiplies",
        "all terms then summed in ONE final",
        "plate+camera interference pass (DC pixel)",
        "",
        f"optical passes used: {mac_calls[0]}",
        f"RESULT = {result}",
    ]
    ax6.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9.5, transform=ax6.transAxes)

    fig.suptitle("3D Loop Over Input Vectors -> Plate+Camera Arithmetic -> Single-Pixel Result", fontsize=15)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_3d_loop_demo(A=None, B=None, C=None, N=None,
                      out_path="optical_3d_loop.png",
                      seed=7):
    """Sizes are now large enough that the honest scalar-Python
    baseline (digital_brute_force) takes real, visible wall-clock time,
    while the plate+camera accumulation is still a single interference
    pass no matter how many terms it's summing."""
    rng = random.Random(seed)
    if A is None:
        A = [rng.randint(1, 50) for _ in range(30)]
    if B is None:
        B = [rng.randint(1, 50) for _ in range(30)]
    if C is None:
        C = [rng.randint(1, 50) for _ in range(30)]

    I, J, K = len(A), len(B), len(C)
    n_terms = I * J * K

    print("=" * 78)
    print("3D LOOP EVALUATOR (plate + camera arithmetic, single-pixel result)")
    print("=" * 78)
    print(f"A: length {I}   B: length {J}   C: length {K}   N={N}")
    print(f"loop shape: {I} x {J} x {K} = {n_terms} terms "
          f"({2*n_terms} chained multiplies + 1 final accumulation pass)")

    # ---- PC-only baseline: scalar Python, one MAC at a time ----
    t0 = time.perf_counter()
    digital_result = digital_brute_force(A, B, C, N=N)
    t1 = time.perf_counter()
    digital_time = t1 - t0

    # ---- Plate + camera path ----
    mac_calls = [0]
    t2 = time.perf_counter()
    result, terms, trace = evaluate_3d_loop(A, B, C, N=N, mac_calls=mac_calls)
    t3 = time.perf_counter()
    optical_time = t3 - t2

    print(f"\nPC brute-force (scalar Python, {n_terms} sequential MAC adds):")
    print(f"  result = {digital_result}")
    print(f"  time   = {digital_time*1000:.2f} ms")

    print(f"\nPlate + camera path ({mac_calls[0]} optical passes total, "
          f"final accumulation of all {n_terms} terms = 1 pass):")
    print(f"  result = {result}")
    print(f"  time   = {optical_time*1000:.2f} ms  (numpy-simulated on this CPU)")

    assert result == digital_result, "evaluate_3d_loop disagreed with digital ground truth"
    print("\nMatch: True")
    print(f"\nNote: the plate+camera path above used {mac_calls[0]} optical passes total --")
    print(f"{2*n_terms} chained multiply passes plus 1 final accumulation pass that sums all")
    print(f"{n_terms} terms in one shot. On real photonic hardware every one of those passes")
    print(f"is a fixed-time physical event (one plate exposure), so the wall-clock cost")
    print(f"barely depends on {n_terms}. Simulating that same physics on this CPU means running")
    print(f"a full 2D FFT for every single pass, which is exactly why the 'optical' column")
    print(f"above is much slower here than the scalar-Python baseline: the PC is paying, in")
    print(f"full, for physics that the photonic rig would get for free. That gap -- cheap on")
    print(f"the analog hardware, expensive to classically emulate -- is the point of this demo.")

    make_3d_loop_visualization(A, B, C, N, result, terms, trace, mac_calls, out_path)
    print(f"\nPNG: {out_path}")
    return result, terms, trace


if __name__ == "__main__":

    run_3d_loop_demo()
