import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
np.set_printoptions(precision=3, suppress=True)
 
 
def build_matrix(rows, cols, seed):
    # HARDWARE: the digital values that will be carried by the
    # entangled laser — nothing physical yet, this is just the data
    # to be loaded onto the beam plate.
    rng = np.random.default_rng(seed)
    return np.round(rng.uniform(0.0, 1.0, (rows, cols)), 2)
 
 
def build_filter(rows, cols, seed):
    # HARDWARE: the pattern of filters that sit over each hole in
    # the beam plate, each one dimming its hole's beam to a value
    # in [0, 1].
    rng = np.random.default_rng(seed)
    return np.round(rng.uniform(0.0, 1.0, (rows, cols)), 2)
 
 
def encode_same_plane(values, filters):
    # HARDWARE: this elementwise product is what happens when the
    # entangled laser shines through the beam plate with holes —
    # each hole carries one value, and the filter over that hole
    # scales the beam passing through it.
    # PLUG IN HERE: entangled laser -> beam plate with holes
    # (each hole = one matrix entry) -> filters over the holes
    # -> resulting beams = encoded plane.
    if values.shape != filters.shape:
        raise ValueError("values and filters must have identical shapes")
    if np.any(filters < 0.0) or np.any(filters > 1.0):
        raise ValueError("filters must be within [0, 1]")
    return values * filters
 
 
def fft2c(x):
    # HARDWARE: stands in for the beams from all the holes
    # interfering with each other downstream and landing on the
    # camera.
    # PLUG IN HERE: open space (or a lens) where the beams from the
    # holes overlap and interfere, camera placed where the
    # interference pattern forms.
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
 
 
def ifft2c(x):
    # HARDWARE: undoes the interference step above, physically the
    # same kind of interference stage run in reverse to reconstruct
    # the correlation pattern for detection.
    # PLUG IN HERE: second interference stage, camera placed at the
    # resulting pattern.
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))
 
 
def make_joint_plane(A, B, separation):
    # HARDWARE: this is where both sets of holes (for A and for B)
    # are placed on the SAME beam plate, side by side.
    # PLUG IN HERE: one beam plate with two hole-groups punched into
    # it — A's holes in one region, B's holes in another region,
    # `separation` holes apart.
    m, p = A.shape
    p2, n = B.shape
    if p != p2:
        raise ValueError(f"A shape {A.shape} and B shape {B.shape} are incompatible")
    width = separation + n
    height = max(m, p)
    plane = np.zeros((height, width), dtype=float)
    A_plane = np.zeros_like(plane)
    B_plane = np.zeros_like(plane)
    A_plane[:m, :p] = A
    B_plane[:p, separation:separation + n] = B
    plane = A_plane + B_plane
    return plane, A_plane, B_plane
 
 
def balanced_joint_transform(A, B, separation):
    # HARDWARE: joint_power is what the camera actually measures —
    # cameras only see brightness (intensity), not the underlying
    # beams directly, so this is the real detector readout.
    # PLUG IN HERE: camera downstream of the beam plate + filters +
    # interference region. Three camera shots needed: A's holes lit
    # alone (F_A), B's holes lit alone (F_B), and both lit together
    # (F_joint) — subtracting the first two from the third in
    # software isolates the interference between A and B.
    joint, A_plane, B_plane = make_joint_plane(A, B, separation)
    F_joint = fft2c(joint)               # HARDWARE: camera shot, both A & B holes lit
    joint_power = np.abs(F_joint) ** 2   # HARDWARE: brightness the camera pixels record
    F_A = fft2c(A_plane)                 # HARDWARE: camera shot, only A's holes lit
    F_B = fft2c(B_plane)                 # HARDWARE: camera shot, only B's holes lit
    self_power = np.abs(F_A) ** 2 + np.abs(F_B) ** 2
    balanced_power = joint_power - self_power
    cross_spectrum = F_A * np.conj(F_B) + np.conj(F_A) * F_B
    cross_correlation = np.real(ifft2c(cross_spectrum))
    return joint, joint_power, balanced_power, cross_correlation
 
 
def optical_mac_reference(A, B):
    """Reference for the correctly decoded shared-index optical channels."""
    # HARDWARE: each outer-product term A[:, j] * B[j, :] is the
    # light from one hole in A's group interfering with one hole in
    # B's group. Summing over j happens for free because the camera
    # just adds up all the light landing on each pixel from every
    # hole pair at once.
    # PLUG IN HERE: beam plate with A's holes and B's holes, filters
    # over every hole, all beams overlapping and interfering in the
    # space before the camera, camera sensor doing the summation by
    # simply recording total brightness per pixel.
    m, p = A.shape
    p2, n = B.shape
    if p != p2:
        raise ValueError("inner dimensions must match")
    channels = np.empty((p, m, n), dtype=float)
    for j in range(p):
        channels[j] = A[:, j:j + 1] * B[j:j + 1, :]
    return channels.sum(axis=0), channels  # HARDWARE: camera readout (summed brightness)
 
 
def plot_matrix(fig, ax, data, title, cmap="viridis", vmin=None, vmax=None, annotate=False):
    image = ax.imshow(
        data,
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    if annotate:
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                ax.text(
                    c, r, f"{data[r, c]:.3f}",
                    ha="center", va="center",
                    color="white", fontsize=8,
                )
    fig.colorbar(image, ax=ax, shrink=0.78)
 
 
def run(
    m=3,
    p=3,
    n=3,
    value_seed_A=7,
    value_seed_B=11,
    filter_seed_A=101,
    filter_seed_B=202,
    separation=8,
    out_path="same_plane_matrix_multiply_full.png",
):
    # ==============================================================
    # 1. DIGITAL REPRESENTATION BEFORE FILTERING
    # ============================================================== 
    values_A = build_matrix(m, p, value_seed_A)
    values_B = build_matrix(p, n, value_seed_B)
 
    # ==============================================================
    # 2. FILTER PLANES, independently constrained to [0, 1]
    #    HARDWARE: the physical filters you'd place over each hole
    #    in the beam plate, right before firing the laser and
    #    taking a camera shot.
    # ============================================================== 
    filters_A = build_filter(m, p, filter_seed_A)
    filters_B = build_filter(p, n, filter_seed_B)
 
    # ==============================================================
    # 3. SAME-PLANE ENCODING
    #    HARDWARE: physically realized by the entangled laser
    #    shining through the beam plate's holes and their filters —
    #    see encode_same_plane().
    # ============================================================== 
    encoded_A = encode_same_plane(values_A, filters_A)
    encoded_B = encode_same_plane(values_B, filters_B)
 
    # ==============================================================
    # 4. DIGITAL RESULT AFTER ENCODING (software ground truth)
    # ============================================================== 
    digital_C = encoded_A @ encoded_B
 
    # ==============================================================
    # 5. JOINT 2-D PLANE AND INTERFERENCE DIAGNOSTIC
    #    HARDWARE: the actual optical bench for this stage —
    #    entangled laser -> beam plate with holes -> filters ->
    #    interference -> camera. See make_joint_plane() and
    #    balanced_joint_transform() for exact plug-in points.
    # ============================================================== 
    joint_plane, joint_power, balanced_power, cross_correlation = balanced_joint_transform(
        encoded_A,
        encoded_B,
        separation,
    )
 
    # ==============================================================
    # 6. SHARED-INDEX OPTICAL MAC REFERENCE
    #    HARDWARE: replace optical_mac_reference() with a real
    #    camera acquisition call to go from simulation to
    #    hardware-in-the-loop.
    # ============================================================== 
    # Each j-channel produces the outer-product contribution:
    # encoded_A[:, j] * encoded_B[j, :].
    optical_C, channel_products = optical_mac_reference(
        encoded_A,
        encoded_B,
    )
 
    absolute_error = np.abs(digital_C - optical_C)
    max_error = np.max(absolute_error)
 
    print("\n========== DIGITAL REPRESENTATION BEFORE ==========")
    print("A values:")
    print(values_A)
    print("\nB values:")
    print(values_B)
 
    print("\n========== FILTER PLANES ==========")
    print("F_A, values in [0, 1]:")
    print(filters_A)
    print("\nF_B, values in [0, 1]:")
    print(filters_B)
 
    print("\n========== SAME-PLANE ENCODING ==========")
    print("T_A = V_A * F_A:")
    print(encoded_A)
    print("\nT_B = V_B * F_B:")
    print(encoded_B)
 
    print("\n========== DIGITAL RESULT AFTER MULTIPLICATION ==========")
    print("C = T_A @ T_B:")
    print(digital_C)
 
    print("\n========== OPTICAL SHARED-INDEX MAC RESULT ==========")
    print(optical_C)
    print("\nMaximum absolute error:")
    print(max_error)
 
    # ==============================================================
    # 7. CLEAR STAGED FIGURE
    # ============================================================== 
    fig, axes = plt.subplots(
        3,
        5,
        figsize=(20, 12),
        constrained_layout=True,
    )
 
    # Row 1: unmistakable digital input before any filtering.
    plot_matrix(
        fig, axes[0, 0], values_A,
        "DIGITAL BEFORE\nA matrix values",
        vmin=0.0, vmax=1.0, annotate=True,
    )
    plot_matrix(
        fig, axes[0, 1], values_B,
        "DIGITAL BEFORE\nB matrix values",
        vmin=0.0, vmax=1.0, annotate=True,
    )
    axes[0, 2].axis("off")
    axes[0, 3].axis("off")
    axes[0, 4].axis("off")
    axes[0, 2].text(
        0.5, 0.5,
        "DIGITAL INPUT\nBEFORE FILTERING",
        ha="center", va="center", fontsize=15,
        transform=axes[0, 2].transAxes,
    )
 
    # Row 2: filters and fields encoded on the same 2-D planes.
    plot_matrix(
        fig, axes[1, 0], filters_A,
        "FILTER PLANE F_A\nvalues in [0, 1]",
        cmap="magma", vmin=0.0, vmax=1.0, annotate=True,
    )
    plot_matrix(
        fig, axes[1, 1], filters_B,
        "FILTER PLANE F_B\nvalues in [0, 1]",
        cmap="magma", vmin=0.0, vmax=1.0, annotate=True,
    )
    plot_matrix(
        fig, axes[1, 2], encoded_A,
        "SAME 2-D PLANE A\nT_A = V_A × F_A",
        vmin=0.0, vmax=1.0, annotate=True,
    )
    plot_matrix(
        fig, axes[1, 3], encoded_B,
        "SAME 2-D PLANE B\nT_B = V_B × F_B",
        vmin=0.0, vmax=1.0, annotate=True,
    )
    axes[1, 4].axis("off")
    axes[1, 4].text(
        0.5, 0.5,
        "FILTER ENCODING\nvalue × filter",
        ha="center", va="center", fontsize=14,
        transform=axes[1, 4].transAxes,
    )
 
    # Row 3: digital after, optical result, error and diagnostics.
    plot_matrix(
        fig, axes[2, 0], digital_C,
        "DIGITAL AFTER\nC = T_A @ T_B",
        cmap="inferno", annotate=True,
    )
    plot_matrix(
        fig, axes[2, 1], optical_C,
        "ACTUAL OPTICAL MAC\nΣ_j T_A[:,j] T_B[j,:]",
        cmap="inferno", annotate=True,
    )
    plot_matrix(
        fig, axes[2, 2], absolute_error,
        "ABSOLUTE ERROR\n|digital − optical|",
        cmap="coolwarm", annotate=True,
    )
    plot_matrix(
        fig, axes[2, 3], joint_plane,
        "SINGLE JOINT 2-D PLANE",
        cmap="viridis",
    )
    plot_matrix(
        fig, axes[2, 4], np.abs(cross_correlation),
        "BALANCED INTERFERENCE\nCROSS-CORRELATION",
        cmap="inferno",
    )
 
    fig.suptitle(
        "Digital Before → Same-Plane Filter Encoding → Digital After → Optical MAC",
        fontsize=17,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
 
    print(f"\nSaved figure to {out_path}")
 
    return {
        "values_A": values_A,
        "values_B": values_B,
        "filters_A": filters_A,
        "filters_B": filters_B,
        "encoded_A": encoded_A,
        "encoded_B": encoded_B,
        "digital_C": digital_C,
        "optical_C": optical_C,
        "absolute_error": absolute_error,
        "joint_plane": joint_plane,
        "joint_power": joint_power,
        "balanced_power": balanced_power,
        "cross_correlation": cross_correlation,
        "channel_products": channel_products,
    }
 
 
if __name__ == "__main__":
    run()
 
