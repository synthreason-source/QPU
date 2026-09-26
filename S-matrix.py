import math
import random
import time
from dataclasses import dataclass

import numpy as np


# ============================================================
# S-MATRIX INTEGER FACTORISATION
#
# Pipeline:
#
#   N
#   |
#   +--> arithmetic feature states
#   |
#   +--> S-matrix
#   |
#   +--> SVD dimensional reduction
#   |
#   +--> reduced candidate scoring
#   |
#   +--> exact divisibility verification
#   |
#   +--> Pollard-Rho fallback
#
# Important:
# The reduced S-matrix is a heuristic search/ranking mechanism.
# Every returned factor is verified exactly with p*q == N.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_RANK = 8
DEFAULT_SAMPLES = 4096
DEFAULT_CANDIDATES = 256

SMALL_PRIMES = (
    2, 3, 5, 7, 11, 13, 17, 19,
    23, 29, 31, 37, 41, 43, 47
)


# ============================================================
# RESULT OBJECT
# ============================================================

@dataclass
class FactorResult:
    n: int
    factor: int | None
    cofactor: int | None
    method: str
    elapsed: float
    verified: bool

    @property
    def success(self):
        return self.verified and self.factor is not None

    def __str__(self):
        if not self.success:
            return (
                f"N={self.n}\n"
                f"method={self.method}\n"
                f"factorization=not found\n"
                f"verified=False\n"
                f"time={self.elapsed:.6f}s"
            )

        return (
            f"N={self.n}\n"
            f"factor={self.factor}\n"
            f"cofactor={self.cofactor}\n"
            f"method={self.method}\n"
            f"verified={self.verified}\n"
            f"time={self.elapsed:.6f}s"
        )


# ============================================================
# BASIC INTEGER UTILITIES
# ============================================================

def integer_sqrt(n):
    return math.isqrt(n)


def is_square(n):
    if n < 0:
        return False

    r = math.isqrt(n)

    return r * r == n


def exact_factor_check(n, p):
    """
    Exact verification.

    No floating-point arithmetic is involved.
    """

    if p <= 1:
        return False

    if n % p != 0:
        return False

    q = n // p

    return p * q == n


def normalize_factor_pair(n, p):
    if p > n // p:
        p = n // p

    q = n // p

    return p, q


# ============================================================
# MILLER-RABIN PRIMALITY TEST
# ============================================================

def is_probable_prime(n):
    if n < 2:
        return False

    for p in SMALL_PRIMES:
        if n == p:
            return True

        if n % p == 0:
            return False

    d = n - 1

    s = 0

    while d % 2 == 0:
        d //= 2
        s += 1

    # Deterministic for 64-bit integers with these bases.
    bases = (
        2, 325, 9375, 28178,
        450775, 9780504, 1795265022
    )

    for a in bases:
        if a % n == 0:
            continue

        x = pow(a, d, n)

        if x == 1 or x == n - 1:
            continue

        witness = True

        for _ in range(s - 1):
            x = pow(x, 2, n)

            if x == n - 1:
                witness = False
                break

        if witness:
            return False

    return True


# ============================================================
# POLLARD-RHO
# ============================================================

def pollard_rho(n, max_iterations=2_000_000):
    """
    Pollard-Rho factor finder.

    Returns a nontrivial factor or None.
    """

    if n % 2 == 0:
        return 2

    if n % 3 == 0:
        return 3

    if n % 5 == 0:
        return 5

    if is_probable_prime(n):
        return n

    for _restart in range(64):

        c = random.randrange(1, n - 1)

        x = random.randrange(2, n - 1)

        y = x

        d = 1

        for _ in range(max_iterations):

            x = (
                (x * x) % n + c
            ) % n

            y = (
                (y * y) % n + c
            ) % n

            y = (
                (y * y) % n + c
            ) % n

            d = math.gcd(
                abs(x - y),
                n
            )

            if d == 1:
                continue

            if d == n:
                break

            return d

    return None


# ============================================================
# COMPLETE POLLARD-RHO FACTORIZATION
# ============================================================

def pollard_factor_recursive(n, factors):
    if n == 1:
        return

    if is_probable_prime(n):
        factors.append(n)
        return

    d = pollard_rho(n)

    if d is None:
        # Retry recursively with a different random trajectory.
        while d is None:
            d = pollard_rho(n)

    pollard_factor_recursive(
        d,
        factors
    )

    pollard_factor_recursive(
        n // d,
        factors
    )


def complete_pollard_factorization(n):
    factors = []

    pollard_factor_recursive(
        n,
        factors
    )

    factors.sort()

    return factors


# ============================================================
# ARITHMETIC FEATURE VECTOR
# ============================================================

def arithmetic_features(n, d):
    """
    Construct a feature vector describing candidate d relative
    to N.

    The important exact quantity is the remainder:

        N mod d

    A valid factor has remainder zero.

    Other features give the S-matrix a structured representation
    of the arithmetic landscape.
    """

    if d <= 0:
        raise ValueError("d must be positive")

    remainder = n % d

    quotient = n // d

    gcd_value = math.gcd(
        n,
        d
    )

    distance_to_division = min(
        remainder,
        d - remainder
    )

    quotient_error = abs(
        n - quotient * d
    )

    # Modular residues at several scales.
    #
    # These are intentionally redundant because the SVD will
    # discover the dominant low-dimensional structure.
    mod_2 = n % 2
    mod_3 = n % 3
    mod_5 = n % 5
    mod_7 = n % 7
    mod_11 = n % 11
    mod_13 = n % 13

    # Logarithmic quantities prevent huge integers from producing
    # numerically useless scales.
    log_d = math.log1p(d)

    log_q = math.log1p(quotient)

    log_remainder = math.log1p(remainder)

    log_gcd = math.log1p(gcd_value)

    log_distance = math.log1p(
        distance_to_division
    )

    # Binary/parity features.
    parity = d & 1

    features = np.array(
        [
            float(remainder),
            float(distance_to_division),
            float(gcd_value),
            float(quotient_error),

            float(mod_2),
            float(mod_3),
            float(mod_5),
            float(mod_7),
            float(mod_11),
            float(mod_13),

            log_d,
            log_q,
            log_remainder,
            log_gcd,
            log_distance,

            float(parity),
        ],
        dtype=np.float64
    )

    return features


# ============================================================
# CANDIDATE DOMAIN
# ============================================================

def generate_candidate_domain(n, samples=DEFAULT_SAMPLES):
    """
    Generate integer candidate divisors in [2, sqrt(N)].

    All candidates remain ordinary Python integers. This avoids
    NumPy integer overflow and NumPy object/logarithm problems for
    large factorization inputs.
    """

    n = int(n)
    samples = max(2, int(samples))

    limit = math.isqrt(n)

    if limit < 2:
        return []

    if limit <= samples:
        return list(
            range(
                2,
                limit + 1
            )
        )

    log_start = math.log(2.0)

    # Convert only for logarithmic positioning.
    #
    # This is safe for normal-sized inputs. For extremely large
    # integers, use the bit-length formulation below.
    bit_length = limit.bit_length()

    log_end = (
        (bit_length - 1) * math.log(2.0)
        + math.log(
            limit / (1 << (bit_length - 1))
        )
    )

    candidates = []

    for i in range(samples):

        fraction = (
            i / (samples - 1)
        )

        log_value = (
            log_start
            + fraction
            * (log_end - log_start)
        )

        # Prevent floating point overflow.
        if log_value >= 709.0:
            candidate = limit
        else:
            candidate = int(
                math.exp(log_value)
            )

        candidate = max(
            2,
            min(
                candidate,
                limit
            )
        )

        candidates.append(
            candidate
        )

    return sorted(
        set(candidates)
    )


# ============================================================
# FEATURE MATRIX
# ============================================================

def build_feature_matrix(
    n,
    candidates
):
    rows = []

    for d in candidates:
        rows.append(
            arithmetic_features(
                n,
                int(d)
            )
        )

    if not rows:
        return np.empty(
            (0, 0),
            dtype=np.float64
        )

    return np.asarray(
        rows,
        dtype=np.float64
    )


# ============================================================
# NORMALIZATION
# ============================================================

def standardize_matrix(X):
    """
    Column-wise standardization.

        X' = (X - mean) / std
    """

    mean = np.mean(
        X,
        axis=0
    )

    std = np.std(
        X,
        axis=0
    )

    std[
        std < 1e-12
    ] = 1.0

    return (
        (X - mean) / std,
        mean,
        std
    )


# ============================================================
# S-MATRIX CONSTRUCTION
# ============================================================

def build_s_matrix(
    feature_matrix
):
    """
    Treat candidate arithmetic feature vectors as the columns
    of the S-matrix.

        S = [f(d1) f(d2) ... f(dk)]
    """

    return feature_matrix.T


# ============================================================
# SVD DIMENSIONAL REDUCTION
# ============================================================

def reduce_s_matrix(
    S,
    rank
):
    if S.size == 0:
        raise ValueError(
            "Cannot reduce an empty S-matrix"
        )

    U, singular_values, Vh = np.linalg.svd(
        S,
        full_matrices=False
    )

    rank = max(
        1,
        min(
            int(rank),
            len(singular_values)
        )
    )

    U_r = U[:, :rank]

    Sigma_r = singular_values[:rank]

    Vh_r = Vh[:rank, :]

    return (
        U_r,
        Sigma_r,
        Vh_r
    )


# ============================================================
# REDUCED COORDINATES
# ============================================================

def reduced_coordinates(
    S,
    U
):
    return (
        np.conjugate(U).T @ S
    )


# ============================================================
# SPECTRAL SCORE
# ============================================================

def calculate_spectral_scores(
    U,
    singular_values,
    Vh
):
    """
    Score candidate columns according to their representation
    in the retained low-dimensional subspace.

    Candidates with strong coherent representation receive
    larger scores.

    This is NOT an exact divisibility test.
    """

    if Vh.size == 0:
        return np.zeros(0)

    weighted = (
        np.abs(Vh)
        * singular_values[:, None]
    )

    scores = np.sum(
        weighted ** 2,
        axis=0
    )

    norm = np.linalg.norm(
        scores
    )

    if norm > 0:
        scores /= norm

    return scores


# ============================================================
# EXACT LOCAL SEARCH
# ============================================================

def exact_neighborhood_search(
    n,
    candidates,
    ranked_indices,
    neighborhood=4
):
    """
    Test candidates selected by the reduced S-matrix.

    Around each spectral candidate we also test nearby integers.
    """

    tested = set()

    for idx in ranked_indices:

        center = int(
            candidates[idx]
        )

        for offset in range(
            -neighborhood,
            neighborhood + 1
        ):

            d = center + offset

            if d < 2:
                continue

            if d in tested:
                continue

            tested.add(d)

            if exact_factor_check(
                n,
                d
            ):
                return d

    return None


# ============================================================
# LOCAL DENSITY SEARCH
# ============================================================

def remainder_score(n, d):
    """
    Exact distance from d to being a divisor.

    Zero means exact divisibility.
    """

    r = n % d

    return min(
        r,
        d - r
    )


def local_remainder_search(
    n,
    candidates,
    top_count
):
    """
    Secondary arithmetic search.

    This is useful when the SVD basis identifies the correct
    scale but does not put the exact divisor directly at the
    top of the spectrum.
    """

    limit = math.isqrt(n)

    if len(candidates) == 0:
        return None

    top_count = min(
        top_count,
        len(candidates)
    )

    # Rank according to closeness to an exact modular zero.
    scored = []

    for d in candidates:
        d = int(d)

        if d > limit:
            continue

        score = remainder_score(
            n,
            d
        )

        scored.append(
            (score, d)
        )

    scored.sort(
        key=lambda item: item[0]
    )

    for _, d in scored[:top_count]:

        if exact_factor_check(
            n,
            d
        ):
            return d

    return None


# ============================================================
# S-MATRIX FACTOR SOLVER
# ============================================================

def smatrix_factor(
    n,
    rank=DEFAULT_RANK,
    samples=DEFAULT_SAMPLES,
    candidate_count=DEFAULT_CANDIDATES
):
    """
    Attempt factorization using the reduced S-matrix.
    """

    start = time.perf_counter()

    # --------------------------------------------------------
    # Trivial cases
    # --------------------------------------------------------

    if n < 2:
        return FactorResult(
            n=n,
            factor=None,
            cofactor=None,
            method="invalid",
            elapsed=time.perf_counter() - start,
            verified=False
        )

    for p in SMALL_PRIMES:

        if n == p:
            return FactorResult(
                n=n,
                factor=p,
                cofactor=1,
                method="small-prime",
                elapsed=time.perf_counter() - start,
                verified=True
            )

        if n % p == 0:

            q = n // p

            verified = (
                p * q == n
            )

            return FactorResult(
                n=n,
                factor=p,
                cofactor=q,
                method="small-prime",
                elapsed=time.perf_counter() - start,
                verified=verified
            )

    if is_probable_prime(n):

        return FactorResult(
            n=n,
            factor=n,
            cofactor=1,
            method="primality-test",
            elapsed=time.perf_counter() - start,
            verified=True
        )

    # --------------------------------------------------------
    # Candidate domain
    # --------------------------------------------------------

    candidates = generate_candidate_domain(
        n,
        samples=samples
    )

    if len(candidates) == 0:

        return FactorResult(
            n=n,
            factor=None,
            cofactor=None,
            method="empty-domain",
            elapsed=time.perf_counter() - start,
            verified=False
        )

    # --------------------------------------------------------
    # Arithmetic feature space
    # --------------------------------------------------------

    feature_matrix = build_feature_matrix(
        n,
        candidates
    )

    feature_matrix, mean, std = (
        standardize_matrix(
            feature_matrix
        )
    )

    # --------------------------------------------------------
    # S-matrix
    # --------------------------------------------------------

    S = build_s_matrix(
        feature_matrix
    )

    # --------------------------------------------------------
    # Dimensional reduction
    # --------------------------------------------------------

    U, singular_values, Vh = (
        reduce_s_matrix(
            S,
            rank
        )
    )

    # --------------------------------------------------------
    # Reduced coordinates
    # --------------------------------------------------------

    _ = reduced_coordinates(
        S,
        U
    )

    # --------------------------------------------------------
    # Spectral candidate ranking
    # --------------------------------------------------------

    scores = calculate_spectral_scores(
        U,
        singular_values,
        Vh
    )

    order = np.argsort(
        scores
    )[::-1]

    top_count = min(
        candidate_count,
        len(order)
    )

    ranked_indices = order[
        :top_count
    ]

    # --------------------------------------------------------
    # Exact verification
    # --------------------------------------------------------

    factor = exact_neighborhood_search(
        n,
        candidates,
        ranked_indices
    )

    if factor is not None:

        factor, cofactor = (
            normalize_factor_pair(
                n,
                factor
            )
        )

        verified = (
            factor * cofactor == n
        )

        if verified:

            return FactorResult(
                n=n,
                factor=factor,
                cofactor=cofactor,
                method="reduced-s-matrix",
                elapsed=time.perf_counter() - start,
                verified=True
            )

    # --------------------------------------------------------
    # Secondary exact remainder search
    # --------------------------------------------------------

    factor = local_remainder_search(
        n,
        candidates,
        top_count
    )

    if factor is not None:

        factor, cofactor = (
            normalize_factor_pair(
                n,
                factor
            )
        )

        verified = (
            factor * cofactor == n
        )

        if verified:

            return FactorResult(
                n=n,
                factor=factor,
                cofactor=cofactor,
                method="reduced-s-matrix-remainder",
                elapsed=time.perf_counter() - start,
                verified=True
            )

    # --------------------------------------------------------
    # Not found
    # --------------------------------------------------------

    return FactorResult(
        n=n,
        factor=None,
        cofactor=None,
        method="reduced-s-matrix-failed",
        elapsed=time.perf_counter() - start,
        verified=False
    )


# ============================================================
# FULL SOLVER WITH POLLARD-RHO FALLBACK
# ============================================================

def factor_integer(
    n,
    rank=DEFAULT_RANK,
    samples=DEFAULT_SAMPLES,
    candidate_count=DEFAULT_CANDIDATES
):
    """
    Main solver.

    1. Try the reduced S-matrix.
    2. Verify the result exactly.
    3. Fall back to Pollard-Rho.
    """

    start = time.perf_counter()

    smatrix_result = smatrix_factor(
        n,
        rank=rank,
        samples=samples,
        candidate_count=candidate_count
    )

    if smatrix_result.success:
        return smatrix_result

    # --------------------------------------------------------
    # Pollard-Rho fallback
    # --------------------------------------------------------

    factor = pollard_rho(n)

    if factor is None:

        return FactorResult(
            n=n,
            factor=None,
            cofactor=None,
            method="pollard-rho-failed",
            elapsed=time.perf_counter() - start,
            verified=False
        )

    cofactor = n // factor

    verified = (
        factor > 1
        and cofactor > 0
        and factor * cofactor == n
    )

    if factor > cofactor:
        factor, cofactor = (
            cofactor,
            factor
        )

    return FactorResult(
        n=n,
        factor=factor,
        cofactor=cofactor,
        method="pollard-rho-fallback",
        elapsed=time.perf_counter() - start,
        verified=verified
    )


# ============================================================
# COMPLETE PRIME FACTORIZATION
# ============================================================

def factor_complete(
    n,
    rank=DEFAULT_RANK,
    samples=DEFAULT_SAMPLES,
    candidate_count=DEFAULT_CANDIDATES
):
    """
    Completely factor N into prime factors.

    The first split uses the S-matrix solver. Subsequent composite
    pieces can again use the same solver before Pollard-Rho.
    """

    if n < 2:
        return []

    if is_probable_prime(n):
        return [n]

    result = factor_integer(
        n,
        rank=rank,
        samples=samples,
        candidate_count=candidate_count
    )

    if not result.success:
        raise RuntimeError(
            "Factorization failed."
        )

    p = result.factor

    q = result.cofactor

    left = factor_complete(
        p,
        rank=rank,
        samples=samples,
        candidate_count=candidate_count
    )

    right = factor_complete(
        q,
        rank=rank,
        samples=samples,
        candidate_count=candidate_count
    )

    return sorted(
        left + right
    )


# ============================================================
# VERIFY COMPLETE FACTORIZATION
# ============================================================

def verify_complete_factorization(
    n,
    factors
):
    if not factors:
        return n == 1

    product = 1

    for f in factors:

        if f < 2:
            return False

        if not is_probable_prime(f):
            return False

        product *= f

    return product == n


# ============================================================
# S-MATRIX DIAGNOSTICS
# ============================================================

def inspect_smatrix(
    n,
    rank=DEFAULT_RANK,
    samples=DEFAULT_SAMPLES
):
    """
    Print the dimensional structure without attempting a full
    factorization.
    """

    candidates = generate_candidate_domain(
        n,
        samples=samples
    )

    features = build_feature_matrix(
        n,
        candidates
    )

    features, _, _ = (
        standardize_matrix(
            features
        )
    )

    S = build_s_matrix(
        features
    )

    U, singular_values, Vh = (
        reduce_s_matrix(
            S,
            rank
        )
    )

    print()
    print("=" * 70)
    print("S-MATRIX DIAGNOSTICS")
    print("=" * 70)

    print(
        "N:",
        n
    )

    print(
        "Candidate states:",
        len(candidates)
    )

    print(
        "Original feature dimensions:",
        S.shape[0]
    )

    print(
        "S-matrix shape:",
        S.shape
    )

    print(
        "Reduced rank:",
        len(singular_values)
    )

    print()
    print("Singular values:")

    for i, value in enumerate(
        singular_values
    ):
        print(
            f"  {i:3d}: {value:.8e}"
        )

    energy = singular_values ** 2

    total = np.sum(
        energy
    )

    if total > 0:

        cumulative = np.cumsum(
            energy / total
        )

        print()
        print("Cumulative captured energy:")

        for i, value in enumerate(
            cumulative
        ):
            print(
                f"  {i:3d}: {value:.8f}"
            )

    return {
        "candidates": candidates,
        "S": S,
        "U": U,
        "singular_values": singular_values,
        "Vh": Vh
    }


# ============================================================
# BENCHMARK
# ============================================================

def benchmark(
    numbers,
    rank=DEFAULT_RANK,
    samples=DEFAULT_SAMPLES,
    candidate_count=DEFAULT_CANDIDATES
):
    print()
    print("=" * 70)
    print("BENCHMARK")
    print("=" * 70)

    for n in numbers:

        print()
        print(
            "N =",
            n
        )

        start = time.perf_counter()

        result = factor_integer(
            n,
            rank=rank,
            samples=samples,
            candidate_count=candidate_count
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            "factor =",
            result.factor
        )

        print(
            "cofactor =",
            result.cofactor
        )

        print(
            "method =",
            result.method
        )

        print(
            "verified =",
            result.verified
        )

        print(
            "time =",
            f"{elapsed:.6f}s"
        )


# ============================================================
# INTERACTIVE SOLVER
# ============================================================

def interactive():
    print()
    print("=" * 70)
    print("REDUCED S-MATRIX INTEGER FACTORISATION")
    print("=" * 70)
    print()
    print("Enter an integer.")
    print("Commands:")
    print("  quit")
    print("  inspect")
    print("  benchmark")
    print()

    while True:

        raw = input(
            "N> "
        ).strip()

        if raw.lower() in (
            "quit",
            "exit",
            "q"
        ):
            break

        if raw.lower() == "inspect":

            raw_n = input(
                "Inspect N> "
            ).strip()

            try:
                n = int(raw_n)

                inspect_smatrix(
                    n
                )

            except Exception as exc:

                print(
                    "Error:",
                    exc
                )

            continue

        if raw.lower() == "benchmark":

            numbers = [
                8051,
                10403,
                100003 * 100019,
                1000003 * 1000033,
            ]

            benchmark(
                numbers
            )

            continue

        try:
            n = int(raw)

        except ValueError:

            print(
                "Please enter an integer."
            )

            continue

        if n < 2:

            print(
                "N must be >= 2."
            )

            continue

        print()
        print(
            "Solving:",
            n
        )

        result = factor_integer(
            n
        )

        print()
        print(result)

        # ----------------------------------------------------
        # Complete factorization
        # ----------------------------------------------------

        if result.success:

            try:

                factors = factor_complete(
                    n
                )

                valid = (
                    verify_complete_factorization(
                        n,
                        factors
                    )
                )

                print()
                print(
                    "Prime factors:",
                    " * ".join(
                        str(x)
                        for x in factors
                    )
                )

                print(
                    "Complete verification:",
                    valid
                )

            except Exception as exc:

                print(
                    "Complete factorization error:",
                    exc
                )

        print()


# ============================================================
# DEMONSTRATION
# ============================================================

def demonstration():

    examples = [
        8051,
        10403,
        1009 * 1013,
        10007 * 10009,
        100003 * 100019,
    ]

    print("=" * 70)
    print("S-MATRIX FACTORIZATION DEMONSTRATION")
    print("=" * 70)

    for n in examples:

        print()
        print(
            "------------------------------------------------------------"
        )

        print(
            "N =",
            n
        )

        result = factor_integer(
            n,
            rank=DEFAULT_RANK,
            samples=DEFAULT_SAMPLES,
            candidate_count=DEFAULT_CANDIDATES
        )

        print(
            "Factor:",
            result.factor
        )

        print(
            "Cofactor:",
            result.cofactor
        )

        print(
            "Method:",
            result.method
        )

        print(
            "Exact verification:",
            result.verified
        )

        if result.success:

            print(
                "Check:",
                result.factor,
                "*",
                result.cofactor,
                "=",
                result.factor * result.cofactor
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":


    print()

    interactive()
