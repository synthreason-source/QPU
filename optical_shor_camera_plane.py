import math
import random
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
# MODULAR REDUCTION -- still explicitly NOT an optical operation. The
# plate + camera can sum gated values via interference; it has no
# mechanism for "subtract N until below N." That stays digital control
# logic between optical_multiply calls.
# =====================================================================

def mod_mul_optical(x: int, y: int, N: int) -> int:
    product = optical_multiply(x % N, y % N)
    return product % N  # digital reduction, not optical


def modexp_optical(a: int, e: int, N: int, mac_calls: list[int] | None = None) -> int:
    result = 1 % N
    base = a % N
    while e > 0:
        if e & 1:
            result = mod_mul_optical(result, base, N)
            if mac_calls is not None:
                mac_calls[0] += 1
        base = mod_mul_optical(base, base, N)
        if mac_calls is not None:
            mac_calls[0] += 1
        e >>= 1
    return result


# =====================================================================
# PERIOD FINDING VIA INTERFERENCE -- the sequence f(x) = a^x mod N is
# still built one valve/camera pass at a time (every point costs one
# modexp_optical call, itself built from the plate+camera multiply
# above). A single 1D FFT of that whole sequence is then a second,
# independent interference spectrum -- the period is read off a peak
# in it via continued-fraction phase estimation, exactly the classical
# post-processing step used after a real quantum measurement in Shor's
# algorithm.
# =====================================================================

def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def find_order_interference(a: int, N: int, mac_calls: list[int], Q_cap: int = 1 << 15):
    Q_needed = next_pow2(N * N)
    Q = min(Q_needed, Q_cap)
    precision_reduced = Q < Q_needed

    seq = np.empty(Q, dtype=float)
    for x in range(Q):
        seq[x] = modexp_optical(a, x, N, mac_calls)

    spectrum = np.abs(np.fft.fft(seq)) ** 2
    order_by_strength = np.argsort(spectrum[1:])[::-1] + 1

    for k in order_by_strength[:16]:
        phase = Fraction(int(k), Q).limit_denominator(N)
        r_candidate = phase.denominator
        if r_candidate > 1 and pow(a, r_candidate, N) == 1:
            return r_candidate, seq, spectrum, int(k), Q, precision_reduced

    return None, seq, spectrum, None, Q, precision_reduced


# =====================================================================
# SHOR'S CLASSICAL POST-PROCESSING (gcd extraction) -- unchanged; this
# part was always classical, even in a real quantum implementation.
# =====================================================================

def try_factor(N: int, max_attempts: int = 25, rng: random.Random | None = None, Q_cap: int = 1 << 15) -> dict:
    if N < 4:
        raise ValueError("N must be >= 4")
  
    rng = rng or random.Random()
    total_mac_calls = 0
    history = []

    for attempt in range(1, max_attempts + 1):
        a = rng.randrange(2, N)
        g = math.gcd(a, N)
        if g != 1:
            factors = (g, N // g)
            history.append({"attempt": attempt, "a": a, "outcome": f"gcd(a,N)={g} directly (lucky)"})
            return {"success": True, "factors": factors, "attempts": attempt,
                    "method": "direct gcd (a shares a factor with N)",
                    "mac_calls": total_mac_calls, "history": history, "spectral": None}

        mac_calls = [0]
        r, seq, spectrum, peak_bin, Q, precision_reduced = find_order_interference(a, N, mac_calls, Q_cap=Q_cap)
        total_mac_calls += mac_calls[0]
        spectral_info = {"a": a, "seq": seq, "spectrum": spectrum, "peak_bin": peak_bin,
                          "Q": Q, "precision_reduced": precision_reduced}

        if r is None:
            note = " (reduced spectral precision, Q capped)" if precision_reduced else ""
            history.append({"attempt": attempt, "a": a, "outcome": f"no peak yielded a valid order{note}"})
            continue
        if r % 2 != 0:
            history.append({"attempt": attempt, "a": a, "outcome": f"order r={r} is odd, unusable"})
            continue

        x = modexp_optical(a, r // 2, N, mac_calls)
        if x == N - 1:
            history.append({"attempt": attempt, "a": a, "outcome": f"order r={r}, a^(r/2)=-1 mod N, trivial"})
            continue

        f1 = math.gcd(x - 1, N)
        f2 = math.gcd(x + 1, N)
        candidate = None
        for f in (f1, f2):
            if 1 < f < N:
                candidate = f
                break

        if candidate is None:
            history.append({"attempt": attempt, "a": a, "outcome": f"order r={r}, gcd extraction gave trivial factors"})
            continue

        other = N // candidate
        verified = (candidate * other == N)
        history.append({"attempt": attempt, "a": a, "outcome": f"order r={r} (interference peak bin {peak_bin}/{Q}), factor {candidate} found, verified={verified}"})
        if verified:
            return {"success": True, "factors": (candidate, other), "attempts": attempt,
                    "method": f"plate+camera Shor post-processing, a={a}, r={r}",
                    "mac_calls": total_mac_calls, "history": history, "spectral": spectral_info}

    return {"success": False, "factors": None, "attempts": max_attempts,
            "method": "exhausted attempts", "mac_calls": total_mac_calls, "history": history, "spectral": None}


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


def make_visualization(N, spectral, x_example, y_example, result, out_path):
    fig = plt.figure(figsize=(22, 15), layout="constrained")
    grid = fig.add_gridspec(3, 4)

    product, optics = optical_multiply(x_example, y_example, return_optics=True)

    ax1 = fig.add_subplot(grid[0, 0])
    plot_matrix(fig, ax1, optics["holes_plate"], f"PLATE HOLES (bits of x={x_example})\nstrictly binary, one-shot",
                cmap="Greys_r", vmin=0, vmax=1, annotate=True)

    ax2 = fig.add_subplot(grid[0, 1])
    plot_matrix(fig, ax2, optics["values_plate"], f"ENTANGLED LASER (y={y_example} << i)\nsame 2D grid as the holes",
                cmap="viridis", annotate=(optics["values_plate"].size <= 64))

    ax3 = fig.add_subplot(grid[0, 2])
    plot_matrix(fig, ax3, optics["filtered_plate"], "2D PLANE FILTER OUTPUT\n(laser x holes, elementwise)",
                cmap="plasma", annotate=(optics["filtered_plate"].size <= 64))

    ax4 = fig.add_subplot(grid[0, 3])
    plot_matrix(fig, ax4, np.log1p(optics["intensity"]), "CAMERA IMAGE\nlog(1+|FFT(filtered plate)|^2)",
                cmap="magma", mark=optics["dc_index"])

    ax5 = fig.add_subplot(grid[1, 0:1])
    ax5.axis("off")
    dc_val = optics["field"][optics["dc_index"]]
    lines = [
        "MULTIPLY, VIA PLATE + CAMERA", "",
        f"x = {x_example}  (2D hole plate)",
        f"y = {y_example}  (2D laser values plate)",
        "", "filter: laser x holes (elementwise)",
        "camera: single 2D FFT (Fraunhofer)",
        f"DC pixel (center) = {dc_val.real:.3f}{dc_val.imag:+.3f}j",
        f"optical result (Re[DC], rounded): {product}",
        f"digital check:  {x_example * y_example}",
        "", "Modular reduction (%N) happens AFTER",
        "this camera readout, in digital control",
        "logic -- the optics has no mod operation.",
    ]
    ax5.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9.5, transform=ax5.transAxes)

    a_used = spectral["a"]
    seq = spectral["seq"]
    spectrum = spectral["spectrum"]
    peak_bin = spectral["peak_bin"]
    Q = spectral["Q"]

    ax6 = fig.add_subplot(grid[1, 1:3])
    show_n = min(len(seq), 512)
    ax6.plot(range(show_n), seq[:show_n], marker="o", markersize=2.5, linewidth=0.7)
    ax6.set_title(f"f(x) = {a_used}^x mod {N}, x=0..{Q-1}  (each point = one plate+camera modexp pass)")
    ax6.set_xlabel("x")
    ax6.set_ylabel("a^x mod N")
    ax6.grid(alpha=0.25)

    ax7 = fig.add_subplot(grid[1, 3])
    show_bins = min(len(spectrum), 512)
    ax7.plot(range(show_bins), spectrum[:show_bins], marker="o", markersize=2.5, linewidth=0.7, color="crimson")
    if peak_bin is not None and peak_bin < show_bins:
        ax7.axvline(peak_bin, color="black", linestyle="--", linewidth=1,
                    label=f"peak used (bin {peak_bin})")
        ax7.legend(fontsize=7)
    ax7.set_title("PERIOD SPECTRUM of f(x)\n(1D interference calc, its own\ncamera-style FFT readout)")
    ax7.set_xlabel("frequency bin")
    ax7.grid(alpha=0.25)

    ax8 = fig.add_subplot(grid[2, 0:2])
    ax8.axis("off")
    precision_note = "reduced (Q capped)" if spectral["precision_reduced"] else "full (Q >= N^2)"
    summary = [
        "RESULT", "",
        f"N = {N}",
        f"Success: {result['success']}",
        f"Factors: {result['factors']}",
        f"Method: {result['method']}",
        f"Attempts used: {result['attempts']}",
        f"Optical MAC calls: {result['mac_calls']}",
        f"Spectral precision: {precision_note}",
        "",
        "Every multiply now goes: 2D hole plate +",
        "entangled-laser values plate -> elementwise",
        "plane filter -> single FFT propagation to a",
        "camera -> MAC sum read off the DC pixel of",
        "the resulting interference image. Period",
        "extraction is a second, 1D interference",
        "readout (continued fractions off its peak).",
    ]
    ax8.text(0.0, 1.0, "\n".join(summary), va="top", family="monospace", fontsize=8.5, transform=ax8.transAxes)

    fig.suptitle("Plate Holes + Entangled Laser -> 2D Plane Filter -> Camera-Image Interference Readout", fontsize=15)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(N=9, seed=0, max_attempts=25, out_path="optical_shor_camera_plane.png", Q_cap=1 << 15):
    rng = random.Random(seed)

    print("=" * 78)
    print("PLATE + CAMERA FACTORIZATION (interference-based multiply and period-finding)")
    print("=" * 78)
    print(f"N = {N}")

    # sanity-check the multiply primitive against plain arithmetic first
    check_x, check_y = 13, 47
    optical_product = optical_multiply(check_x, check_y)
    assert optical_product == check_x * check_y, "optical_multiply disagreed with digital ground truth"
    print(f"Multiply primitive check: {check_x} * {check_y} = {optical_product} (matches digital: {optical_product == check_x*check_y})")

    result = try_factor(N, max_attempts=max_attempts, rng=rng, Q_cap=Q_cap)

    print("\n---- attempt history ----")
    for h in result["history"]:
        print(f"  attempt {h['attempt']:>2}: a={h['a']:>5}  {h['outcome']}")

    print("\n---- result ----")
    print(f"Success:          {result['success']}")
    print(f"Factors:          {result['factors']}")
    print(f"Method:           {result['method']}")
    print(f"Attempts used:    {result['attempts']}")
    print(f"Optical MAC calls: {result['mac_calls']}")

    if result["success"]:
        f1, f2 = result["factors"]
        assert f1 * f2 == N, "internal inconsistency: reported factors do not multiply back to N"
        print(f"Independent check: {f1} * {f2} = {f1*f2} == N: {f1*f2 == N}")

    spectral = result["spectral"]
    if spectral is None:
        a_demo = rng.randrange(2, N)
        mac_calls_demo = [0]
        _, seq, spectrum, peak_bin, Q, precision_reduced = find_order_interference(a_demo, N, mac_calls_demo, Q_cap=Q_cap)
        spectral = {"a": a_demo, "seq": seq, "spectrum": spectrum, "peak_bin": peak_bin,
                    "Q": Q, "precision_reduced": precision_reduced}

    make_visualization(N, spectral, check_x, check_y, result, out_path)
    print(f"\nPNG: {out_path}")
    return result


if __name__ == "__main__":
    run()