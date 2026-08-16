import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)


def build_matrix(rows, cols, seed):
    rng = np.random.default_rng(seed)
    return np.round(rng.uniform(0.0, 1.0, (rows, cols)), 2)


def build_filter(rows, cols, seed):
    rng = np.random.default_rng(seed)
    return np.round(rng.uniform(0.0, 1.0, (rows, cols)), 2)


def encode_same_plane(values, filters):
    if values.shape != filters.shape:
        raise ValueError("values and filters must have identical shapes")
    if np.any(filters < 0.0) or np.any(filters > 1.0):
        raise ValueError("filters must be within [0, 1]")
    return values * filters


def fft2c(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))


def ifft2c(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))


def make_joint_plane(A, B, separation):
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
    joint, A_plane, B_plane = make_joint_plane(A, B, separation)
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
    # ============================================================== 
    filters_A = build_filter(m, p, filter_seed_A)
    filters_B = build_filter(p, n, filter_seed_B)

    # ==============================================================
    # 3. SAME-PLANE ENCODING
    # ============================================================== 
    encoded_A = encode_same_plane(values_A, filters_A)
    encoded_B = encode_same_plane(values_B, filters_B)

    # ==============================================================
    # 4. DIGITAL RESULT AFTER ENCODING
    # ============================================================== 
    digital_C = encoded_A @ encoded_B

    # ==============================================================
    # 5. JOINT 2-D PLANE AND INTERFERENCE DIAGNOSTIC
    # ============================================================== 
    joint_plane, joint_power, balanced_power, cross_correlation = balanced_joint_transform(
        encoded_A,
        encoded_B,
        separation,
    )

    # ==============================================================
    # 6. SHARED-INDEX OPTICAL MAC REFERENCE
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
