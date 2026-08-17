import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)


def build_values(rows, cols, seed, p_one=0.5):
    # HARDWARE: the raw digital data before anything touches a hole
    # on the beam plate -- this never exists physically, it's just
    # bits sitting in software. Strictly 0 or 1, same as the holes
    # they'll become.
    rng = np.random.default_rng(seed)
    return (rng.uniform(0.0, 1.0, (rows, cols)) < p_one).astype(float)


def to_holes(values):
    # HARDWARE: physically realizing the digital bits as holes on
    # the plate -- a 1 becomes an open hole (beam passes), a 0
    # becomes a shut hole (beam blocked). Since the digital data is
    # already binary, this is a direct one-to-one placement, not a
    # threshold/rounding decision.
    return values.copy()


def fft2c(x):
    # HARDWARE: beams from all open holes interfering downstream
    # and landing on the camera.
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))


def ifft2c(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))


def make_joint_plane(A, B):
    # HARDWARE: ONE beam plate, fully packed. A's holes and B's
    # holes sit immediately next to each other -- no gap, no second
    # region, no dead space between them. Every hole on the plate is
    # either an A hole or a B hole; there is no padding column.
    # width = p + n exactly (not p + separation + n).
    m, p = A.shape
    p2, n = B.shape
    if p != p2:
        raise ValueError(f"A shape {A.shape} and B shape {B.shape} are incompatible")
    width = p + n
    height = max(m, p)
    plane = np.zeros((height, width), dtype=float)
    A_plane = np.zeros_like(plane)
    B_plane = np.zeros_like(plane)
    A_plane[:m, :p] = A
    B_plane[:p, p:p + n] = B
    plane = A_plane + B_plane
    return plane, A_plane, B_plane


def balanced_joint_transform(A, B):
    joint, A_plane, B_plane = make_joint_plane(A, B)
    F_joint = fft2c(joint)
    joint_power = np.abs(F_joint) ** 2
    F_A = fft2c(A_plane)
    F_B = fft2c(B_plane)
    self_power = np.abs(F_A) ** 2 + np.abs(F_B) ** 2
    balanced_power = joint_power - self_power
    cross_spectrum = F_A * np.conj(F_B) + np.conj(F_A) * F_B
    cross_correlation = np.real(ifft2c(cross_spectrum))
    return joint, joint_power, balanced_power, cross_correlation


def optical_mac_reference(A, B):
    """Reference for the correctly decoded shared-index optical channels."""
    m, p = A.shape
    p2, n = B.shape
    if p != p2:
        raise ValueError("inner dimensions must match")
    channels = np.empty((p, m, n), dtype=float)
    for j in range(p):
        channels[j] = A[:, j:j + 1] * B[j:j + 1, :]
    return channels.sum(axis=0), channels


def plot_matrix(fig, ax, data, title, cmap="viridis", vmin=None, vmax=None, annotate=False):
    image = ax.imshow(
        data, origin="lower", interpolation="nearest",
        cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
    )
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    if annotate:
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                ax.text(c, r, f"{data[r, c]:.0f}", ha="center", va="center",
                         color="red", fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.78)


def run(
    m=3, p=3, n=3,
    seed_A=7, seed_B=11,
    p_one=0.5,
    out_path="single_plane_binary_matrix_multiply.png",
):
    # ==============================================================
    # 0. DIGITAL BEFORE -- raw binary data (0/1), nothing physical
    #    yet. This is what exists purely in software prior to any
    #    hole being opened or shut.
    # ==============================================================
    values_A = build_values(m, p, seed_A, p_one)
    values_B = build_values(p, n, seed_B, p_one)

    # ==============================================================
    # 1. ONE SINGLE, FULLY-PACKED BINARY HOLE PATTERN -- the same
    #    0/1 bits placed directly onto the plate, A's holes and B's
    #    holes touching, no gap, no separate value vs. filter layer
    #    downstream of this point. Every hole is open (1) or shut (0).
    # ==============================================================
    A = to_holes(values_A)
    B = to_holes(values_B)
    joint_plane, A_plane, B_plane = make_joint_plane(A, B)

    # ==============================================================
    # 2. DIGITAL GROUND TRUTH (plain binary matrix multiply)
    # ==============================================================
    digital_C = A @ B

    # ==============================================================
    # 3. INTERFERENCE DIAGNOSTIC FROM THE SAME SINGLE PLANE
    # ==============================================================
    _, joint_power, balanced_power, cross_correlation = balanced_joint_transform(A, B)

    # ==============================================================
    # 4. SHARED-INDEX OPTICAL MAC REFERENCE
    # ==============================================================
    optical_C, channel_products = optical_mac_reference(A, B)

    absolute_error = np.abs(digital_C - optical_C)
    max_error = np.max(absolute_error)

    print("\n========== DIGITAL BEFORE (raw 0/1 bits, nothing physical yet) ==========")
    print("A values:\n", values_A)
    print("\nB values:\n", values_B)

    print("\n========== SINGLE BINARY PLANE (same bits, placed as holes) ==========")
    print("A holes (0=shut, 1=open):\n", A)
    print("\nB holes (0=shut, 1=open):\n", B)
    print(f"\nFraction of A holes open: {A.mean():.2f}")
    print(f"Fraction of B holes open: {B.mean():.2f}")

    print("\n========== DIGITAL RESULT ==========")
    print("C = A @ B:\n", digital_C)

    print("\n========== OPTICAL SHARED-INDEX MAC RESULT ==========")
    print(optical_C)
    print("\nMaximum absolute error:")
    print(max_error)

    # ==============================================================
    # 5. FIGURE
    # ==============================================================
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)

    plot_matrix(fig, axes[0, 0], values_A, "DIGITAL BEFORE\nA values (0/1)",
                cmap="Greys_r", vmin=0, vmax=1, annotate=True)
    plot_matrix(fig, axes[0, 1], values_B, "DIGITAL BEFORE\nB values (0/1)",
                cmap="Greys_r", vmin=0, vmax=1, annotate=True)
    axes[0, 2].axis("off")
    axes[0, 3].axis("off")
    axes[0, 2].text(0.5, 0.5, "same bits,\nplaced directly\nas holes",
                     ha="center", va="center", fontsize=14, transform=axes[0, 2].transAxes)

    plot_matrix(fig, axes[1, 0], A, "A holes\n(0/1 only)",
                cmap="Greys_r", vmin=0, vmax=1, annotate=True)
    plot_matrix(fig, axes[1, 1], B, "B holes\n(0/1 only)",
                cmap="Greys_r", vmin=0, vmax=1, annotate=True)
    plot_matrix(fig, axes[1, 2], joint_plane, "ONE SINGLE PLANE\nA + B holes, no gap",
                cmap="Greys_r", vmin=0, vmax=1)
    axes[1, 3].axis("off")
    axes[1, 3].text(0.5, 0.5, "ONE PLATE\nONE SHOT\nALL BINARY\nNO DEAD SPACE",
                     ha="center", va="center", fontsize=13, transform=axes[1, 3].transAxes)

    plot_matrix(fig, axes[2, 0], digital_C, "DIGITAL AFTER\nC = A @ B",
                cmap="inferno", annotate=True)
    plot_matrix(fig, axes[2, 1], optical_C, "OPTICAL MAC\nSUM_j A[:,j] B[j,:]",
                cmap="inferno", annotate=True)
    plot_matrix(fig, axes[2, 2], absolute_error, "ABSOLUTE ERROR\n|digital - optical|",
                cmap="coolwarm", annotate=True)
    plot_matrix(fig, axes[2, 3], np.abs(cross_correlation),
                "INTERFERENCE\nCROSS-CORRELATION", cmap="inferno")

    fig.suptitle(
        "Digital Before (0/1) -> One Binary Plane -> Digital After -> Optical MAC",
        fontsize=16,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved figure to {out_path}")

    return {
        "values_A": values_A,
        "values_B": values_B,
        "A": A, "B": B,
        "joint_plane": joint_plane,
        "digital_C": digital_C,
        "optical_C": optical_C,
        "absolute_error": absolute_error,
        "joint_power": joint_power,
        "balanced_power": balanced_power,
        "cross_correlation": cross_correlation,
        "channel_products": channel_products,
    }


if __name__ == "__main__":
    run()
