import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


np.set_printoptions(
    precision=3,
    suppress=True,
)


LAMBDA = 1.0
K = 2.0 * np.pi / LAMBDA


def build_matrix(
    rows,
    cols,
    seed,
    minimum=0.0,
    maximum=1.0,
):
    rng = np.random.default_rng(seed)

    return np.round(
        rng.uniform(
            minimum,
            maximum,
            size=(rows, cols),
        ),
        2,
    )


def apply_filter(matrix, threshold):
    return np.where(
        matrix > threshold,
        matrix,
        0.0,
    )


def huygens_propagate(
    src_x,
    src_y,
    src_field,
    obs_x,
    obs_y,
    z,
    wavelength=LAMBDA,
    direction=+1,
    source_area=1.0,
):
    """
    Scalar spherical-wave propagation.

    direction=+1:
        exp(+i k r): forward propagation.

    direction=-1:
        exp(-i k r): adjoint/backward propagation.
    """
    k = 2.0 * np.pi / wavelength

    dx = obs_x[:, None] - src_x[None, :]
    dy = obs_y[:, None] - src_y[None, :]

    r = np.sqrt(
        dx**2
        + dy**2
        + z**2
    )

    kernel = (
        np.exp(1j * direction * k * r)
        / r
    )

    return source_area * (
        kernel @ src_field
    )


def normalize_intensity(intensity):
    maximum = np.max(intensity)

    if maximum <= 0.0:
        return intensity.copy()

    return intensity / maximum


def make_grid(extent):
    lo, hi, step = extent

    values = np.arange(
        lo,
        hi + 0.5 * step,
        step,
        dtype=float,
    )

    X, Y = np.meshgrid(
        values,
        values,
        indexing="ij",
    )

    return X, Y


def matrix_multiply_hologram(
    A,
    B,
    pitch=8.0,
    d=150.0,
    hologram_extent=(-20.0, 80.0, 2.0),
    output_extent=None,
    wavelength=LAMBDA,
):
    """
    Simulate matrix multiplication through coherent wave propagation.

    For:

        A.shape = (m, p)
        B.shape = (p, n)

    the expected result is:

        C = A @ B

    Each output element C[i, k] is represented by a target point.
    Its target amplitude is the matrix-product value:

        C[i, k] = sum_j A[i, j] B[j, k]

    The hologram is synthesized from those output point sources.
    """

    m, p = A.shape
    p_b, n = B.shape

    if p != p_b:
        raise ValueError(
            "Matrix dimensions are incompatible: "
            f"A has shape {A.shape}, "
            f"B has shape {B.shape}."
        )

    if output_extent is None:
        output_extent = (
            0.0,
            (max(m, n) - 1) * pitch,
            1.0,
        )

    C_expected = A @ B

    # --------------------------------------------------------------
    # Output target coordinates
    # --------------------------------------------------------------

    output_indices_i = np.arange(
        m,
        dtype=float,
    )

    output_indices_k = np.arange(
        n,
        dtype=float,
    )

    Ii, Kk = np.meshgrid(
        output_indices_i,
        output_indices_k,
        indexing="ij",
    )

    target_x = Ii.ravel() * pitch
    target_y = Kk.ravel() * pitch
    target_amplitude = C_expected.ravel()

    # Only nonzero matrix-product terms contribute target sources.
    active = target_amplitude != 0.0

    # --------------------------------------------------------------
    # Hologram sampling plane
    # --------------------------------------------------------------

    U, V = make_grid(
        hologram_extent
    )

    Uf = U.ravel()
    Vf = V.ravel()

    hologram_pixel_area = (
        hologram_extent[2] ** 2
    )

    print(
        f"A shape: {A.shape}"
    )

    print(
        f"B shape: {B.shape}"
    )

    print(
        f"Expected C=A@B shape: {C_expected.shape}"
    )

    print(
        f"Nonzero output elements: "
        f"{np.count_nonzero(active)} / {m * n}"
    )

    print(
        "Synthesizing hologram from matrix-product "
        "target sources..."
    )

    # Adjoint/backward propagation to the hologram.
    hologram_field = huygens_propagate(
        src_x=target_x[active],
        src_y=target_y[active],
        src_field=target_amplitude[active].astype(complex),
        obs_x=Uf,
        obs_y=Vf,
        z=d,
        wavelength=wavelength,
        direction=-1,
        source_area=1.0,
    )

    # --------------------------------------------------------------
    # Forward reconstruction
    # --------------------------------------------------------------

    Xo, Yo = make_grid(
        output_extent
    )

    Xof = Xo.ravel()
    Yof = Yo.ravel()

    reconstructed_field = huygens_propagate(
        src_x=Uf,
        src_y=Vf,
        src_field=hologram_field,
        obs_x=Xof,
        obs_y=Yof,
        z=d,
        wavelength=wavelength,
        direction=+1,
        source_area=hologram_pixel_area,
    )

    reconstructed_field = reconstructed_field.reshape(
        Xo.shape
    )

    reconstructed_intensity = (
        np.abs(reconstructed_field) ** 2
    )

    # --------------------------------------------------------------
    # Sample reconstructed values at matrix-product coordinates
    # --------------------------------------------------------------

    reconstructed_at_targets = huygens_propagate(
        src_x=Uf,
        src_y=Vf,
        src_field=hologram_field,
        obs_x=target_x,
        obs_y=target_y,
        z=d,
        wavelength=wavelength,
        direction=+1,
        source_area=hologram_pixel_area,
    )

    actual_target_intensity = (
        np.abs(reconstructed_at_targets) ** 2
    )

    expected_target_intensity = (
        target_amplitude ** 2
    )

    expected_active = expected_target_intensity[active]
    actual_active = actual_target_intensity[active]

    expected_normalized = (
        expected_active / np.max(expected_active)
    )

    if np.max(actual_active) > 0:
        actual_normalized = (
            actual_active / np.max(actual_active)
        )
    else:
        actual_normalized = actual_active

    if (
        len(expected_normalized) > 1
        and np.std(expected_normalized) > 0
        and np.std(actual_normalized) > 0
    ):
        correlation = np.corrcoef(
            expected_normalized,
            actual_normalized,
        )[0, 1]
    else:
        correlation = np.nan

    print(
        "Correlation between expected matrix-product "
        f"intensity and reconstructed intensity: {correlation:.4f}"
    )

    # --------------------------------------------------------------
    # Build ideal output image
    # --------------------------------------------------------------

    ideal_output = np.zeros_like(
        reconstructed_intensity
    )

    output_lo = output_extent[0]
    output_step = output_extent[2]

    for index in range(m * n):
        ix = int(
            round(
                (target_x[index] - output_lo)
                / output_step
            )
        )

        iy = int(
            round(
                (target_y[index] - output_lo)
                / output_step
            )
        )

        if (
            0 <= ix < ideal_output.shape[0]
            and 0 <= iy < ideal_output.shape[1]
        ):
            ideal_output[ix, iy] = (
                target_amplitude[index] ** 2
            )

    return {
        "expected": C_expected,
        "hologram_field": hologram_field.reshape(U.shape),
        "reconstructed_field": reconstructed_field,
        "reconstructed_intensity": reconstructed_intensity,
        "ideal_output": ideal_output,
        "target_x": target_x,
        "target_y": target_y,
        "active": active,
        "correlation": correlation,
    }


def run(
    m=4,
    p=5,
    n=3,
    filter_a=0.1,
    filter_b=0.2,
    pitch=8.0,
    d=150.0,
    hologram_extent=(-20.0, 80.0, 2.0),
    output_extent=None,
    seed_a=7,
    seed_b=11,
    out_path="matrix_multiply_hologram.png",
):
    """
    Generate two matrices, filter them, multiply them, and simulate
    the matrix-product output as holographically reconstructed spots.
    """

    A_before = build_matrix(
        rows=m,
        cols=p,
        seed=seed_a,
    )

    B_before = build_matrix(
        rows=p,
        cols=n,
        seed=seed_b,
    )

    A_after = apply_filter(
        A_before,
        filter_a,
    )

    B_after = apply_filter(
        B_before,
        filter_b,
    )

    result = matrix_multiply_hologram(
        A=A_after,
        B=B_after,
        pitch=pitch,
        d=d,
        hologram_extent=hologram_extent,
        output_extent=output_extent,
    )

    C_expected = result["expected"]

    actual_intensity = result[
        "reconstructed_intensity"
    ]

    ideal_output = result[
        "ideal_output"
    ]

    target_x = result["target_x"]
    target_y = result["target_y"]
    active = result["active"]
    correlation = result["correlation"]

    print("\nA before filtering:")
    print(A_before)

    print("\nA after filtering:")
    print(A_after)

    print("\nB before filtering:")
    print(B_before)

    print("\nB after filtering:")
    print(B_after)

    print("\nExpected matrix product C=A@B:")
    print(C_expected)

    # --------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------

    if output_extent is None:
        output_extent = (
            0.0,
            (max(m, n) - 1) * pitch,
            1.0,
        )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9),
        constrained_layout=True,
    )

    # A before
    ax = axes[0, 0]

    image = ax.imshow(
        A_before,
        cmap="viridis",
        origin="lower",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    ax.set_title("A before filtering")
    ax.set_xlabel("j")
    ax.set_ylabel("i")

    fig.colorbar(
        image,
        ax=ax,
        shrink=0.8,
    )

    # A after
    ax = axes[0, 1]

    image = ax.imshow(
        A_after,
        cmap="viridis",
        origin="lower",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    ax.set_title("A after filtering")
    ax.set_xlabel("j")
    ax.set_ylabel("i")

    fig.colorbar(
        image,
        ax=ax,
        shrink=0.8,
    )

    # B after
    ax = axes[0, 2]

    image = ax.imshow(
        B_after,
        cmap="viridis",
        origin="lower",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    ax.set_title("B after filtering")
    ax.set_xlabel("k")
    ax.set_ylabel("j")

    fig.colorbar(
        image,
        ax=ax,
        shrink=0.8,
    )

    # Expected numerical matrix multiplication
    ax = axes[1, 0]

    image = ax.imshow(
        C_expected,
        cmap="inferno",
        origin="lower",
        interpolation="nearest",
    )

    ax.set_title("Simulation\nNumerical C = A @ B")
    ax.set_xlabel("k")
    ax.set_ylabel("i")

    for i in range(m):
        for k in range(n):
            ax.text(
                k,
                i,
                f"{C_expected[i, k]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )

    fig.colorbar(
        image,
        ax=ax,
        shrink=0.8,
    )

    # Ideal optical output
    ax = axes[1, 1]

    image = ax.imshow(
        normalize_intensity(ideal_output).T,
        cmap="inferno",
        origin="lower",
        extent=[
            output_extent[0],
            output_extent[1],
            output_extent[0],
            output_extent[1],
        ],
        interpolation="nearest",
    )

    ax.scatter(
        target_x[active],
        target_y[active],
        s=40,
        facecolors="none",
        edgecolors="cyan",
        label="Expected output spots",
    )

    ax.set_title("Ideal optical output\n|A @ B|²")
    ax.set_xlabel("output x")
    ax.set_ylabel("output y")
    ax.legend(fontsize=8)

    fig.colorbar(
        image,
        ax=ax,
        shrink=0.8,
    )

    # Actual optical reconstruction
    ax = axes[0, 2]

    # Replace the B panel with the actual reconstruction in the
    # bottom-right panel below. This panel remains B for clarity.
    ax = axes[1, 2]

    image = ax.imshow(
        normalize_intensity(actual_intensity).T,
        cmap="inferno",
        origin="lower",
        extent=[
            output_extent[0],
            output_extent[1],
            output_extent[0],
            output_extent[1],
        ],
        interpolation="nearest",
    )

    ax.scatter(
        target_x[active],
        target_y[active],
        s=40,
        facecolors="none",
        edgecolors="cyan",
        label="Expected output spots",
    )

    ax.set_title(
        "Actual holographic output\n"
        f"correlation = {correlation:.4f}"
    )

    ax.set_xlabel("output x")
    ax.set_ylabel("output y")
    ax.legend(fontsize=8)

    fig.colorbar(
        image,
        ax=ax,
        shrink=0.8,
    )

    fig.suptitle(
        "Matrix Multiplication by Coherent Wave Interference",
        fontsize=16,
    )

    fig.savefig(
        out_path,
        dpi=140,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\nSaved figure to {out_path}"
    )

    return {
        "A_before": A_before,
        "A_after": A_after,
        "B_before": B_before,
        "B_after": B_after,
        "C_expected": C_expected,
        **result,
    }


if __name__ == "__main__":
    run()