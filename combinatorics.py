"""
class RealOpticalHardware(OpticalHardwareInterface):

    def upload_amplitude_field(self, amplitude_field):
        # Send the same 2D amplitude field to the coherent modulator.
        pass

    def upload_binary_holes(self, holes):
        # Send the same-shaped {0, 1} aperture to the binary modulator.
        pass

    def expose(self):
        # Trigger laser illumination and camera acquisition.
        pass

    def read_camera(self):
        # Return the acquired 2D camera intensity frame.
        pass

    def reset(self):
        # Clear or prepare the modulators for the next pass.
        pass

"""





import math
import random
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


np.set_printoptions(precision=3, suppress=True)


# =====================================================================
# MODEL
# =====================================================================
#
# The simulated hardware consists of:
#
#   1. Coherent amplitude/value field
#      A complex-valued 2D field E[y, x].
#
#   2. Binary 2D hole plate
#      H[y, x] ∈ {0, 1}.
#
#      H[y, x] = 0 -> closed/opaque hole
#      H[y, x] = 1 -> open/transparent hole
#
#   3. Physical 2D amplitude filter
#
#          E_filtered[y, x] = E[y, x] * H[y, x]
#
#      This is the only operation performed by the plate.
#
#   4. Lens/Fraunhofer propagation
#      A lens produces the Fourier-plane field. The FFT below is a
#      numerical model of that physical propagation.
#
#   5. Camera
#      The camera records optical intensity:
#
#          I[u, v] = |F{E_filtered}[u, v]|^2
#
#   6. DC camera pixel
#      The central Fourier-plane pixel corresponds to zero spatial
#      frequency. Its complex amplitude is proportional to:
#
#          sum_y sum_x E_filtered[y, x]
#
#      For nonnegative, phase-aligned real amplitudes, the sum can be
#      recovered from sqrt(DC intensity), after calibration.
#
# IMPORTANT:
# The software FFT is not an optical operation in real hardware. It is
# only the numerical simulation of the lens and propagation.
# =====================================================================


# =====================================================================
# ARRAY AND FOURIER UTILITIES
# =====================================================================

def fft2c(field: np.ndarray) -> np.ndarray:
    """Centered 2D Fourier transform."""
    field = np.asarray(field)
    return np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(field)
        )
    )


def ifft2c(field: np.ndarray) -> np.ndarray:
    """Centered inverse 2D Fourier transform."""
    field = np.asarray(field)
    return np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(field)
        )
    )


def to_square_plate(vector: np.ndarray) -> np.ndarray:
    """Place a 1D vector on a square 2D physical pixel grid.

    The unused padded sites contain zero amplitude. The corresponding
    holes must also be zero, so padding cannot contribute optically.
    """
    vector = np.asarray(vector)

    if vector.ndim != 1:
        raise ValueError("vector must be one-dimensional")

    n = vector.size
    side = int(np.ceil(np.sqrt(max(n, 1))))

    plate = np.zeros((side, side), dtype=vector.dtype)
    plate.flat[:n] = vector

    return plate


def vector_and_binary_holes(vector: np.ndarray):
    """Construct a same-shaped 2D amplitude plate and binary hole plate."""
    vector = np.asarray(vector)

    amplitude_plate = to_square_plate(vector)

    holes_plate = np.zeros_like(amplitude_plate, dtype=float)
    holes_plate.flat[:vector.size] = 1.0

    return amplitude_plate, holes_plate


# =====================================================================
# PHYSICAL 2D BINARY AMPLITUDE FILTER
# =====================================================================

def binary_2d_amplitude_filter(
    amplitude_field: np.ndarray,
    holes: np.ndarray,
) -> np.ndarray:
    """Apply a strictly binary 2D amplitude mask.

    Physical interpretation:

        coherent laser field
            -> binary amplitude plate
            -> filtered optical field

    Both inputs describe exactly the same physical 2D pixel grid.
    """

    amplitude_field = np.asarray(amplitude_field, dtype=complex)
    holes = np.asarray(holes, dtype=float)

    if amplitude_field.ndim != 2:
        raise ValueError("amplitude_field must be a 2D array")

    if holes.ndim != 2:
        raise ValueError("holes must be a 2D array")

    if amplitude_field.shape != holes.shape:
        raise ValueError(
            "amplitude_field and holes must have exactly the same shape"
        )

    if not np.all(np.isin(holes, [0.0, 1.0])):
        raise ValueError(
            "holes must be strictly binary: 0.0 or 1.0"
        )

    # ================================================================
    # PHYSICAL PLATE OPERATION
    #
    # Every amplitude pixel is paired with exactly one binary hole:
    #
    #       E_filtered[y, x] = E[y, x] * H[y, x]
    #
    # No partial transmission is permitted.
    # ================================================================
    return amplitude_field * holes


# Backward-compatible name from the previous version.
entangled_laser_illuminate = binary_2d_amplitude_filter
encode_same_plane = binary_2d_amplitude_filter


# =====================================================================
# OPTICAL PROPAGATION AND CAMERA
# =====================================================================

def propagate_to_camera(filtered_field: np.ndarray):
    """Model lens propagation and camera intensity measurement.

    Returns:
        camera_field:
            Complex Fourier-plane optical field.

        camera_intensity:
            Intensity measured by the camera sensor.
    """

    # ================================================================
    # PHYSICAL LENS OPERATION
    #
    # A lens at the focal-plane configuration performs a Fourier
    # transform of the complex field transmitted through the plate.
    #
    # The FFT is the numerical model of that optical propagation.
    # ================================================================
    camera_field = fft2c(filtered_field)

    # ================================================================
    # PHYSICAL CAMERA OPERATION
    #
    # A conventional camera measures optical power/intensity and does
    # not directly measure complex amplitude:
    #
    #       I = |camera_field|^2
    # ================================================================
    camera_intensity = np.abs(camera_field) ** 2

    return camera_field, camera_intensity


def camera_dc_pixel(camera_field: np.ndarray):
    """Return the center/zero-order Fourier-plane pixel."""

    cy = camera_field.shape[0] // 2
    cx = camera_field.shape[1] // 2

    return camera_field[cy, cx], (cy, cx)


def dc_camera_readout(
    camera_field: np.ndarray,
    camera_intensity: np.ndarray,
):
    """Extract the DC camera readout.

    The complex field at DC is:

        F_DC = sum of all filtered plate values

    A normal intensity camera measures:

        I_DC = |F_DC|^2

    Therefore, for phase-aligned nonnegative fields:

        estimated_sum = sqrt(I_DC)
    """

    dc_field, dc_index = camera_dc_pixel(camera_field)
    dc_intensity = float(camera_intensity[dc_index])

    return {
        "dc_index": dc_index,
        "dc_field": dc_field,
        "dc_intensity": dc_intensity,
        "dc_amplitude_magnitude": math.sqrt(dc_intensity),
    }


def run_optical_pass(
    amplitude_field: np.ndarray,
    holes: np.ndarray,
):
    """Run one complete simulated physical optical pass."""

    filtered_field = binary_2d_amplitude_filter(
        amplitude_field,
        holes,
    )

    camera_field, camera_intensity = propagate_to_camera(
        filtered_field
    )

    readout = dc_camera_readout(
        camera_field,
        camera_intensity,
    )

    return {
        "amplitude_field": np.asarray(amplitude_field),
        "holes": np.asarray(holes),
        "filtered_field": filtered_field,
        "camera_field": camera_field,
        "camera_intensity": camera_intensity,
        **readout,
    }


# =====================================================================
# HARDWARE INTERFACE
# =====================================================================

class OpticalHardwareInterface:
    """Abstract interface for a real optical implementation.

    A physical implementation would replace the simulation methods with
    hardware-specific code for:

      - an amplitude/phase modulator for the value field;
      - a DMD, binary SLM, or fabricated aperture for the holes;
      - a coherent laser;
      - a Fourier-transform lens;
      - a camera or coherent detector.
    """

    def upload_amplitude_field(self, amplitude_field: np.ndarray):
        raise NotImplementedError

    def upload_binary_holes(self, holes: np.ndarray):
        raise NotImplementedError

    def expose(self):
        raise NotImplementedError

    def read_camera(self) -> np.ndarray:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class SimulatedOpticalHardware(OpticalHardwareInterface):
    """CPU simulation of the plate, lens, and camera."""

    def __init__(self):
        self.amplitude_field = None
        self.holes = None
        self.filtered_field = None
        self.camera_field = None
        self.camera_intensity = None

    def upload_amplitude_field(
        self,
        amplitude_field: np.ndarray,
    ):
        # Physical hardware:
        # Send the coherent amplitude/value pattern to the modulator.
        self.amplitude_field = np.asarray(
            amplitude_field,
            dtype=complex,
        )

    def upload_binary_holes(
        self,
        holes: np.ndarray,
    ):
        # Physical hardware:
        # Send the strictly binary 2D aperture pattern to the DMD,
        # binary SLM, or physical mask.
        holes = np.asarray(holes, dtype=float)

        if holes.ndim != 2:
            raise ValueError("holes must be a 2D array")

        if not np.all(np.isin(holes, [0.0, 1.0])):
            raise ValueError(
                "physical holes must contain only 0.0 or 1.0"
            )

        self.holes = holes

    def expose(self):
        """Trigger one simulated laser exposure and camera frame."""

        if self.amplitude_field is None:
            raise RuntimeError(
                "upload_amplitude_field() must be called first"
            )

        if self.holes is None:
            raise RuntimeError(
                "upload_binary_holes() must be called first"
            )

        if self.amplitude_field.shape != self.holes.shape:
            raise ValueError(
                "amplitude field and holes must have the same shape"
            )

        # ================================================================
        # PHYSICAL 2D FILTER
        #
        # The laser field and hole field occupy the same physical pixels.
        # This is the only operation performed at the plate.
        # ================================================================
        self.filtered_field = binary_2d_amplitude_filter(
            self.amplitude_field,
            self.holes,
        )

        # ================================================================
        # PHYSICAL FOURIER LENS
        #
        # The FFT numerically represents the field at the lens focal plane.
        # ================================================================
        self.camera_field, self.camera_intensity = (
            propagate_to_camera(self.filtered_field)
        )

    def read_camera(self) -> np.ndarray:
        """Read the simulated camera intensity image."""
        if self.camera_intensity is None:
            raise RuntimeError(
                "expose() must be called before read_camera()"
            )

        return self.camera_intensity.copy()

    def read_dc(self):
        """Read the central DC location from the simulated detector."""
        if self.camera_field is None:
            raise RuntimeError(
                "expose() must be called before read_dc()"
            )

        return dc_camera_readout(
            self.camera_field,
            self.camera_intensity,
        )

    def reset(self):
        self.amplitude_field = None
        self.holes = None
        self.filtered_field = None
        self.camera_field = None
        self.camera_intensity = None


def hardware_plate_pass(
    amplitude_field: np.ndarray,
    holes: np.ndarray,
    hardware=None,
):
    """Execute one amplitude-field/binary-hole hardware pass."""

    if hardware is None:
        hardware = SimulatedOpticalHardware()

    # Classical controller -> value-field modulator.
    hardware.upload_amplitude_field(amplitude_field)

    # Classical controller -> binary 2D hole plate.
    hardware.upload_binary_holes(holes)

    # One physical event:
    #
    #   laser illumination
    #       + binary plate transmission
    #       + lens Fourier propagation
    #       + camera integration
    #
    hardware.expose()

    camera_intensity = hardware.read_camera()
    readout = hardware.read_dc()

    optics = {
        "amplitude_field": np.asarray(amplitude_field),
        "holes": np.asarray(holes),
        "filtered_field": hardware.filtered_field,
        "camera_field": hardware.camera_field,
        "camera_intensity": camera_intensity,
        **readout,
    }

    hardware.reset()

    return optics


# =====================================================================
# INTEGER MULTIPLICATION
# =====================================================================
#
# x * y = sum_i bit_i(x) * (y << i)
#
# The same physical 2D site contains:
#
#   amplitude_field[y, x] -> y << i
#   holes[y, x]           -> bit_i(x)
#
# The binary plate performs multiplication by each bit. The Fourier-plane
# DC interference readout sums the transmitted partial products.
# =====================================================================

def optical_multiply(
    x: int,
    y: int,
    return_optics: bool = False,
    hardware=None,
):
    """Multiply two nonnegative integers with one 2D optical pass."""

    if x < 0 or y < 0:
        raise ValueError(
            "optical_multiply supports unsigned magnitudes only"
        )

    if x == 0 or y == 0:
        if return_optics:
            empty = np.zeros((1, 1), dtype=float)

            return 0, {
                "amplitude_field": empty.astype(complex),
                "holes": empty,
                "filtered_field": empty.astype(complex),
                "camera_field": empty.astype(complex),
                "camera_intensity": empty,
                "dc_index": (0, 0),
                "dc_field": 0j,
                "dc_intensity": 0.0,
                "dc_amplitude_magnitude": 0.0,
            }

        return 0

    width = x.bit_length()

    # Classical control prepares the binary hole values.
    bits = np.array(
        [(x >> i) & 1 for i in range(width)],
        dtype=float,
    )

    # Classical control prepares the value amplitude at each matching site.
    shifted_y = np.array(
        [y * (1 << i) for i in range(width)],
        dtype=float,
    )

    # Both arrays are embedded onto the same genuine 2D grid.
    amplitude_field, holes = vector_and_binary_holes(shifted_y)

    # Replace only the valid data sites with x's binary bits.
    holes.flat[:width] = bits

    optics = hardware_plate_pass(
        amplitude_field,
        holes,
        hardware=hardware,
    )

    # For this simulation, all values are real, nonnegative, and phase
    # aligned, so the DC amplitude equals the desired integer sum.
    product = int(round(optics["dc_field"].real))

    if return_optics:
        return product, optics

    return product


# =====================================================================
# OPTICAL SUM OF MANY VALUES
# =====================================================================

def optical_add_many(
    values,
    N: int | None = None,
    return_optics: bool = False,
    hardware=None,
):
    """Sum many values in one 2D binary-filter/camera pass."""

    arr = np.asarray(values, dtype=float)

    if arr.ndim != 1:
        raise ValueError("values must be one-dimensional")

    if arr.size == 0:
        if return_optics:
            empty = np.zeros((1, 1), dtype=float)

            return 0, {
                "amplitude_field": empty.astype(complex),
                "holes": empty,
                "filtered_field": empty.astype(complex),
                "camera_field": empty.astype(complex),
                "camera_intensity": empty,
                "dc_index": (0, 0),
                "dc_field": 0j,
                "dc_intensity": 0.0,
                "dc_amplitude_magnitude": 0.0,
            }

        return 0

    # Create the value amplitude plane and its matching binary holes plane.
    amplitude_field, holes = vector_and_binary_holes(arr)

    # All valid data pixels are open. Padding pixels remain closed.
    holes.flat[:arr.size] = 1.0

    optics = hardware_plate_pass(
        amplitude_field,
        holes,
        hardware=hardware,
    )

    total = int(round(optics["dc_field"].real))

    if N is not None:
        total %= N

    if return_optics:
        return total, optics

    return total


# =====================================================================
# 3D LOOP EVALUATOR
# =====================================================================
#
# Computes:
#
#       sum_i sum_j sum_k A_i * B_j * C_k
#
# The loop itself remains classical control logic. Every multiplication and
# the final accumulation are routed through the same 2D binary-filter
# hardware model.
# =====================================================================

def evaluate_3d_loop(
    A,
    B,
    C,
    N: int | None = None,
    mac_calls: list[int] | None = None,
    hardware=None,
):
    """Evaluate a triple loop using repeated optical passes."""

    if hardware is None:
        hardware = SimulatedOpticalHardware()

    terms = []
    trace = []

    for i, a in enumerate(A):
        for j, b in enumerate(B):

            # Optical pass 1:
            # Same 2D amplitude grid, binary 2D holes, one DC readout.
            ab = optical_multiply(
                int(a),
                int(b),
                hardware=hardware,
            )

            if mac_calls is not None:
                mac_calls[0] += 1

            if N is not None:
                ab %= N

            for k, c in enumerate(C):

                # Optical pass 2:
                # Same hardware model for (a*b)*c.
                term = optical_multiply(
                    ab,
                    int(c),
                    hardware=hardware,
                )

                if mac_calls is not None:
                    mac_calls[0] += 1

                if N is not None:
                    term %= N

                terms.append(term)
                trace.append(
                    (i, j, k, a, b, c, term)
                )

    # Final optical accumulation:
    # Every term is encoded on one amplitude site, all valid sites are open,
    # and the DC camera pixel reads their coherent sum.
    result = optical_add_many(
        terms,
        N=N,
        hardware=hardware,
    )

    if mac_calls is not None:
        mac_calls[0] += 1

    return result, terms, trace


# =====================================================================
# DIGITAL BASELINE
# =====================================================================

def digital_brute_force(
    A,
    B,
    C,
    N: int | None = None,
):
    """Literal scalar-Python baseline."""

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


# =====================================================================
# VISUALIZATION
# =====================================================================

def plot_matrix(
    fig,
    ax,
    data,
    title,
    cmap="viridis",
    vmin=None,
    vmax=None,
    annotate=False,
    mark=None,
):
    image = ax.imshow(
        data,
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )

    ax.set_title(title, fontsize=10)

    if annotate and data.size <= 64:
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                value = data[r, c]

                if np.iscomplexobj(data):
                    label = f"{value.real:.0f}"
                else:
                    label = f"{value:.0f}"

                ax.text(
                    c,
                    r,
                    label,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=7,
                )

    if mark is not None:
        my, mx = mark
        ax.plot(
            mx,
            my,
            marker="x",
            color="lime",
            markersize=12,
            markeredgewidth=2,
        )

    fig.colorbar(image, ax=ax, shrink=0.75)


def make_optical_multiply_visualization(
    x,
    y,
    product,
    optics,
    out_path,
):
    """Visualize one complete 2D binary-filter multiply."""

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
        layout="constrained",
    )

    plot_matrix(
        fig,
        axes[0, 0],
        optics["amplitude_field"].real,
        "2D amplitude/value field",
        cmap="viridis",
        annotate=optics["amplitude_field"].size <= 64,
    )

    plot_matrix(
        fig,
        axes[0, 1],
        optics["holes"],
        "2D binary holes\n0 = shut, 1 = open",
        cmap="gray",
        vmin=0,
        vmax=1,
        annotate=optics["holes"].size <= 64,
    )

    plot_matrix(
        fig,
        axes[0, 2],
        optics["filtered_field"].real,
        "Filtered field\namplitude × binary holes",
        cmap="plasma",
        annotate=optics["filtered_field"].size <= 64,
    )

    plot_matrix(
        fig,
        axes[1, 0],
        np.abs(optics["camera_field"]),
        "Fourier-plane field magnitude",
        cmap="magma",
    )

    plot_matrix(
        fig,
        axes[1, 1],
        optics["camera_intensity"],
        "Camera intensity |field|²",
        cmap="inferno",
        mark=optics["dc_index"],
    )

    axes[1, 2].axis("off")

    lines = [
        "OPTICAL MULTIPLY",
        "",
        f"x = {x}",
        f"y = {y}",
        f"x × y = {product}",
        "",
        f"DC index: {optics['dc_index']}",
        f"DC field: {optics['dc_field']}",
        f"DC intensity: {optics['dc_intensity']:.3f}",
        "",
        "The center camera pixel is the",
        "zero-order Fourier component.",
    ]

    axes[1, 2].text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        family="monospace",
        fontsize=10,
        transform=axes[1, 2].transAxes,
    )

    fig.suptitle(
        "Same 2D Amplitude Field + Binary 2D Hole Filter",
        fontsize=15,
    )

    fig.savefig(
        out_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def make_3d_loop_visualization(
    A,
    B,
    C,
    N,
    result,
    terms,
    trace,
    mac_calls,
    out_path,
):
    fig = plt.figure(
        figsize=(20, 11),
        layout="constrained",
    )

    grid = fig.add_gridspec(2, 3)

    I = len(A)
    J = len(B)
    K = len(C)

    k0 = 0

    if K > 0:
        slice_ij = np.array([
            [
                next(
                    t
                    for ii, jj, kk, a, b, c, t in trace
                    if ii == i and jj == j and kk == k0
                )
                for j in range(J)
            ]
            for i in range(I)
        ])
    else:
        slice_ij = np.zeros((I, J))

    ax1 = fig.add_subplot(grid[0, 0])
    plot_matrix(
        fig,
        ax1,
        np.array(A, dtype=float).reshape(1, -1),
        f"INPUT A (length {I})",
        cmap="viridis",
        annotate=I <= 16,
    )

    ax2 = fig.add_subplot(grid[0, 1])
    plot_matrix(
        fig,
        ax2,
        np.array(B, dtype=float).reshape(1, -1),
        f"INPUT B (length {J})",
        cmap="viridis",
        annotate=J <= 16,
    )

    ax3 = fig.add_subplot(grid[0, 2])
    plot_matrix(
        fig,
        ax3,
        np.array(C, dtype=float).reshape(1, -1),
        f"INPUT C (length {K})",
        cmap="viridis",
        annotate=K <= 16,
    )

    ax4 = fig.add_subplot(grid[1, 0])
    plot_matrix(
        fig,
        ax4,
        slice_ij,
        f"LOOP TERMS AᵢBⱼCₖ\nslice at k={k0}",
        cmap="plasma",
        annotate=I * J <= 64,
    )

    ax5 = fig.add_subplot(grid[1, 1])

    term_amplitude, term_holes = vector_and_binary_holes(
        np.array(terms, dtype=float)
    )

    plot_matrix(
        fig,
        ax5,
        term_amplitude.real,
        f"FINAL AMPLITUDE PLATE\n{len(terms)} terms",
        cmap="plasma",
        annotate=term_amplitude.size <= 64,
    )

    ax6 = fig.add_subplot(grid[1, 2])
    ax6.axis("off")

    lines = [
        "3D LOOP -> 2D OPTICS -> RESULT",
        "",
        f"A length: {I}",
        f"B length: {J}",
        f"C length: {K}",
        f"N: {N}",
        "",
        f"loop shape: {I} × {J} × {K}",
        f"terms: {I * J * K}",
        "",
        "Each multiply uses:",
        "  same 2D amplitude grid",
        "  same-shaped binary holes",
        "  one Fourier propagation",
        "  one DC camera readout",
        "",
        f"optical passes: {mac_calls[0]}",
        f"result: {result}",
    ]

    ax6.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        family="monospace",
        fontsize=9.5,
        transform=ax6.transAxes,
    )

    fig.suptitle(
        "3D Loop with Same-Grid Amplitude and Binary 2D Optical Filtering",
        fontsize=15,
    )

    fig.savefig(
        out_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# DEMOS
# =====================================================================

def run_single_multiply_demo(
    x=13,
    y=17,
    out_path="optical_multiply.png",
):
    """Run and visualize one optical multiplication."""

    product, optics = optical_multiply(
        x,
        y,
        return_optics=True,
    )

    expected = x * y

    print("=" * 78)
    print("SINGLE OPTICAL MULTIPLY")
    print("=" * 78)
    print(f"x: {x}")
    print(f"y: {y}")
    print(f"optical result: {product}")
    print(f"digital result: {expected}")
    print(f"match: {product == expected}")
    print(f"2D plate shape: {optics['holes'].shape}")
    print(f"DC camera index: {optics['dc_index']}")
    print(f"DC intensity: {optics['dc_intensity']:.3f}")

    make_optical_multiply_visualization(
        x,
        y,
        product,
        optics,
        out_path,
    )

    print(f"PNG: {out_path}")

    return product, optics


def run_3d_loop_demo(
    A=None,
    B=None,
    C=None,
    N=None,
    out_path="optical_3d_loop.png",
    seed=7,
):
    """Run the 3D loop and compare with scalar Python."""

    rng = random.Random(seed)

    if A is None:
        A = [rng.randint(1, 50) for _ in range(8)]

    if B is None:
        B = [rng.randint(1, 50) for _ in range(8)]

    if C is None:
        C = [rng.randint(1, 50) for _ in range(8)]

    I = len(A)
    J = len(B)
    K = len(C)

    n_terms = I * J * K

    print("=" * 78)
    print("3D LOOP EVALUATOR")
    print("Same 2D amplitude field + strictly binary 2D hole filter")
    print("=" * 78)
    print(f"A length: {I}")
    print(f"B length: {J}")
    print(f"C length: {K}")
    print(f"N: {N}")
    print(f"loop shape: {I} × {J} × {K}")
    print(f"number of terms: {n_terms}")

    # ---------------------------------------------------------------
    # DIGITAL BASELINE
    # ---------------------------------------------------------------
    t0 = time.perf_counter()

    digital_result = digital_brute_force(
        A,
        B,
        C,
        N=N,
    )

    digital_time = time.perf_counter() - t0

    # ---------------------------------------------------------------
    # OPTICAL-SIMULATION PATH
    # ---------------------------------------------------------------
    hardware = SimulatedOpticalHardware()
    mac_calls = [0]

    t1 = time.perf_counter()

    optical_result, terms, trace = evaluate_3d_loop(
        A,
        B,
        C,
        N=N,
        mac_calls=mac_calls,
        hardware=hardware,
    )

    optical_time = time.perf_counter() - t1

    print()
    print("DIGITAL SCALAR BASELINE")
    print(f"result: {digital_result}")
    print(f"time: {digital_time * 1000:.3f} ms")

    print()
    print("OPTICAL HARDWARE SIMULATION")
    print(f"result: {optical_result}")
    print(f"time: {optical_time * 1000:.3f} ms")
    print(f"simulated optical passes: {mac_calls[0]}")

    print()
    print(f"match: {optical_result == digital_result}")

    if optical_result != digital_result:
        raise AssertionError(
            "optical result disagreed with digital result"
        )

    make_3d_loop_visualization(
        A,
        B,
        C,
        N,
        optical_result,
        terms,
        trace,
        mac_calls,
        out_path,
    )

    print(f"PNG: {out_path}")

    return optical_result, terms, trace


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":

    # ---------------------------------------------------------------
    # Small visual multiplication example.
    # ---------------------------------------------------------------
    run_single_multiply_demo(
        x=13,
        y=17,
        out_path="optical_multiply.png",
    )

    # ---------------------------------------------------------------
    # 3D loop example.
    #
    # These sizes are intentionally modest because this version performs
    # a full 2D FFT for every simulated optical pass. Real hardware would
    # perform the plate/lens/camera event physically rather than executing
    # a CPU FFT for every pass.
    # ---------------------------------------------------------------
    run_3d_loop_demo(
        A=[3, 5, 7, 11],
        B=[2, 4, 6, 8],
        C=[1, 3, 5, 7],
        N=None,
        out_path="optical_3d_loop.png",
    )
