"""
Applying entropy to break the N-dependence of the hole count.

Every earlier version of this rig was DETERMINISTIC trial division:
test a candidate against every prime factor up to sqrt(N). That's
provably correct, but the column count (holes) is forced to grow
with N -- there's no way around it, because you have to rule out
every possible factor to be CERTAIN.

Miller-Rabin gives up certainty for a knob you can turn: instead of
testing against every possible factor, test against k RANDOM witness
values (drawn fresh each shot -- this is the entropy). Each witness
that fails to catch n as composite lowers the error probability by
another factor of <= 1/4. After k=20 random witnesses, the chance a
composite slips through as "probably prime" is <= 4**-20 ~ 10**-12 --
smaller than the odds of a hardware bit-flip corrupting the answer
anyway.

The column count k is now a free parameter you choose for your error
budget, completely decoupled from N. That's what makes "stream to
much higher values with far fewer holes" possible: the SAME 20-hole
plate that works at N=10**4 still works, unmodified, at N=10**18 or
N=10**100.

Physical framing: each witness needs log2(n) sequential squaring
rounds (modular exponentiation), so this trades plate WIDTH for plate
REUSE OVER TIME -- k holes, reused ~log2(n) times per witness, k
witnesses per candidate. Spatial holes stay fixed; temporal shots grow
slowly (log N) instead of the deterministic design's column count
growing like sqrt(N)/ln(sqrt(N)).
"""

import random
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sympy


def miller_rabin_witness(n, d, r, a):
    """One witness check -- one 'hole' worth of test, reused over
    r squaring rounds (temporal reuse, not extra spatial holes)."""
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return True  # this witness didn't catch n as composite
    for _ in range(r - 1):
        x = pow(x, 2, n)
        if x == n - 1:
            return True
    return False  # definite proof n is composite


def optical_entropy_prime_test(n, k, rng):
    """
    THE PLATE: k holes, fixed regardless of n. Each hole is filled with
    a fresh random witness value (the entropy) rather than a
    predetermined factor. Error probability <= 4**-k, chosen by k,
    not by N.
    """
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for _ in range(k):
        a = rng.randrange(2, n - 1)
        if not miller_rabin_witness(n, d, r, a):
            return False
    return True


def verify_against_ground_truth(n_trials, magnitude_bits, k, seed):
    rng = random.Random(seed)
    mismatches = 0
    t0 = time.time()
    for _ in range(n_trials):
        n = rng.getrandbits(magnitude_bits) | 1
        got = optical_entropy_prime_test(n, k, rng)
        truth = sympy.isprime(n)
        if got != truth:
            mismatches += 1
    return mismatches, time.time() - t0


if __name__ == "__main__":
    k = 20
    print(f"Fixed hole count k={k} (theoretical error bound <= 4**-{k} ~= {4.0**-k:.2e})\n")

    for bits, label in [(32, "~10^9"), (64, "~1.8e19"), (128, "~3.4e38"), (330, "~100 digits")]:
        mismatches, elapsed = verify_against_ground_truth(
            n_trials=200, magnitude_bits=bits, k=k, seed=bits)
        print(f"magnitude {label:>14} ({bits:>3} bits)  "
              f"200 trials  mismatches={mismatches}  time={elapsed:.2f}s")

    # --- figure: hole count vs N, deterministic vs entropy-based ---
    N_powers = np.arange(3, 61, 3)  # up to N ~ 10^60
    N_vals = 10.0 ** N_powers

    # deterministic column count ~ pi(sqrt(N)) ~ sqrt(N) / ln(sqrt(N))
    sqrtN = np.sqrt(N_vals)
    deterministic_cols = sqrtN / np.log(sqrtN)

    entropy_cols = np.full_like(N_vals, k, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.plot(N_vals, deterministic_cols, "o-", color="crimson",
            label="deterministic: cols ~ pi(sqrt(N)) -- grows with N")
    ax.plot(N_vals, entropy_cols, "o-", color="steelblue",
            label=f"entropy-based: k={k} random witnesses -- fixed, any N")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (candidate magnitude)")
    ax.set_ylabel("holes (factor / witness columns needed)")
    ax.set_title("Entropy decouples hole count from N")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # annotate a few key crossover points
    for n_power in [9, 18, 38, 100]:
        n_val = 10.0 ** n_power
        det = np.sqrt(n_val) / np.log(np.sqrt(n_val))
        ax.annotate(f"N=10^{n_power}\ndeterministic needs\n{det:,.0f} holes",
                    (n_val, det), textcoords="offset points", xytext=(0, 12),
                    fontsize=7, color="crimson", ha="center")

    fig.savefig("entropy_prime_hole_reduction.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved entropy_prime_hole_reduction.png")

"""
Uses the 20-hole entropy plate (optical_entropy_prime_test) to find
and display an actual large prime number, then verifies it against
an independent ground-truth check (sympy.isprime).

The plate never grows past k holes, no matter how many digits the
candidate has -- that's the whole point of the entropy-based design.
"""

import random
import sympy



def find_prime(bits, k=20, seed=None):
    rng = random.Random(seed)
    n = rng.getrandbits(bits) | 1  # odd candidate of the requested bit length
    attempts = 1
    while not optical_entropy_prime_test(n, k, rng):
        n = rng.getrandbits(bits) | 1
        attempts += 1
    return n, attempts


if __name__ == "__main__":
    BITS = 330          # ~100 decimal digits
    K = 20               # holes on the plate -- fixed, independent of BITS
    SEED = 42

    prime, attempts = find_prime(BITS, k=K, seed=SEED)

    print(f"Plate size (holes): {K}")
    print(f"Candidate bit length: {BITS}  (~{len(str(prime))} decimal digits)")
    print(f"Candidates tested before hitting a prime: {attempts}")
    print()
    print(prime)
    print()
    print("Independently verified prime by sympy.isprime:", sympy.isprime(prime))
