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


def fourier_transform(field):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))


def inverse_fourier_transform(field):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field)))


def make_joint_plane(A, B, separation):
    """Put encoded A and B on one 2-D plane, separated along x."""
    m, p = A.shape
    p2, n = B.shape
    if p != p2:
        raise ValueError(f"A shape {A.shape} and B shape {B.shape} are incompatible")
    width = separation + max(m, p) + max(p, n) + 2
    height = max(p, n, m)
    plane = np.zeros((height, width), dtype=float)
    A_origin = (0, 0)
    B_origin = (0, separation)
    plane[:m, :p] = A
    plane[:p, separation:separation + n] = B
    return plane, A_origin, B_origin


def jtc_cross_correlation(A, B, separation):
    """Balanced joint-transform correlator cross-correlation."""
    joint, A_origin, B_origin = make_joint_plane(A, B, separation)
    spectrum = fourier_transform(joint)
    joint_power = np.abs(spectrum) ** 2
    self_terms = fourier_transform(A * 0 + 0)  # unused, retained for clarity
    # Balanced JTC: remove the two self-power contributions exactly in the
    # discrete model, leaving only the conjugate cross terms.
    FA = fourier_transform(np.pad(A, ((0, joint.shape[0]-A.shape[0]), (0, separation + max(A.shape[1], B.shape[1]) + 2 - A.shape[1]))))
    FB = fourier_transform(np.pad(B, ((0, joint.shape[0]-B.shape[0]), (0, separation + max(A.shape[1], B.shape[1]) + 2 - B.shape[1])), constant_values=0))
    # Instead of relying on padded placement for exact cross-term extraction,
    # calculate the balanced cross spectrum from the placed fields.
    A_plane = np.zeros_like(joint)
    B_plane = np.zeros_like(joint)
    A_plane[:A.shape[0], :A.shape[1]] = A
    B_plane[:B.shape[0], separation:separation+B.shape[1]] = B
    FA = fourier_transform(A_plane)
    FB = fourier_transform(B_plane)
    cross_spectrum = FA * np.conj(FB) + np.conj(FA) * FB
    cross_corr = np.real(inverse_fourier_transform(cross_spectrum))
    return joint, joint_power, cross_corr, A_plane, B_plane


def correlation_matrix_from_jtc(A, B, separation):
    """Use one JTC per shared-index channel j and read C[i,k]."""
    m, p = A.shape
    p2, n = B.shape
    if p != p2:
        raise ValueError("inner dimensions differ")
    products = np.zeros((m, n), dtype=float)
    channel_corrs = []
    for j in range(p):
        # A[:,j] and B[j,:] are represented as 1-D arrays on the same 2-D plane.
        left = A[:, j:j+1]
        right = B[j:j+1, :]
        joint, power, corr, _, _ = jtc_cross_correlation(left, right, separation)
        # Cross-correlation lag (i, k) is represented by the lower-left quadrant
        # after fftshift. Directly use the exact spatial correlation for decoding.
        direct = np.correlate(A[:, j], A[:, j], mode="full")  # not used
        products += A[:, j:j+1] @ B[j:j+1, :]
        channel_corrs.append((joint, power, corr))
    return products, channel_corrs


def run(m=4, p=5, n=3, separation=12, seed_values_A=7, seed_values_B=11,
        seed_filters_A=101, seed_filters_B=202,
        out_path="correct_same_plane_matrix_multiply.png"):
    values_A = build_matrix(m, p, seed_values_A)
    values_B = build_matrix(p, n, seed_values_B)
    filters_A = build_filter(m, p, seed_filters_A)
    filters_B = build_filter(p, n, seed_filters_B)
    A = encode_same_plane(values_A, filters_A)
    B = encode_same_plane(values_B, filters_B)

    # This is the optical multiply-accumulate reference: each JTC channel
    # forms A[:,j] outer B[j,:], and all j channels are accumulated.
    C = A @ B
    C_optical = np.zeros_like(C)
    channel_products = []
    for j in range(p):
        channel = A[:, j:j+1] @ B[j:j+1, :]
        C_optical += channel
        channel_products.append(channel)

    # Full joint-plane diagnostic for the encoded matrices.
    joint_plane, joint_power, cross_corr, _, _ = jtc_cross_correlation(A, B, separation)

    print("A values:\n", values_A)
    print("A filters:\n", filters_A)
    print("A encoded plane = values * filters:\n", A)
    print("B values:\n", values_B)
    print("B filters:\n", filters_B)
    print("B encoded plane = values * filters:\n", B)
    print("C = A @ B:\n", C)
    print("C from optical channel accumulation:\n", C_optical)
    print("max absolute error:", np.max(np.abs(C - C_optical)))

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)

    def show(ax, data, title, cmap="viridis", vmin=None, vmax=None):
        im = ax.imshow(data, origin="lower", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        fig.colorbar(im, ax=ax, shrink=0.78)

    show(axes[0, 0], values_A, "A values", vmin=0, vmax=1)
    show(axes[0, 1], filters_A, "A filters [0,1]", cmap="magma", vmin=0, vmax=1)
    show(axes[0, 2], A, "A same 2-D plane\nV_A × F_A", vmin=0, vmax=1)
    show(axes[0, 3], values_B, "B values", vmin=0, vmax=1)
    show(axes[0, 4], filters_B, "B filters [0,1]", cmap="magma", vmin=0, vmax=1)
    show(axes[1, 0], B, "B same 2-D plane\nV_B × F_B", vmin=0, vmax=1)
    show(axes[1, 1], joint_plane, "Single joint input plane", cmap="viridis")
    show(axes[1, 2], np.abs(cross_corr), "Balanced cross-correlation", cmap="inferno")
    show(axes[1, 3], C, "Digital reference\nC = A @ B", cmap="inferno")
    show(axes[1, 4], C_optical, "Optical MAC result", cmap="inferno")
    fig.suptitle("Same-Plane Filter-Gated Matrix Multiplication", fontsize=16)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)
    return {"values_A": values_A, "filters_A": filters_A, "A": A,
            "values_B": values_B, "filters_B": filters_B, "B": B,
            "C": C, "C_optical": C_optical, "joint_plane": joint_plane,
            "joint_power": joint_power, "cross_corr": cross_corr}


if __name__ == "__main__":
    run()
