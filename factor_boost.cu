// factor_boost.cu
//
// Arbitrary-precision prime factorisation using Boost.Multiprecision.
//
// CUDA is used for parallel 64-bit Pollard-Rho searches.
// Arbitrary-precision arithmetic uses boost::multiprecision::cpp_int.
//
// Compile:
//
//   nvcc -O3 -std=c++17 factor_boost.cu -o factor_boost
//
// Run:
//
//   ./factor_boost 360
//   ./factor_boost 18446744073709551617
//   ./factor_boost 340282366920938463463374607431768211507
//   ./factor_boost benchmark
//
// Requirements:
//
//   - CUDA toolkit
//   - Boost headers
//
// On Ubuntu:
//
//   sudo apt install libboost-dev
//
// Notes:
//
//   - cpp_int supports arbitrary precision.
//   - CUDA Pollard-Rho acceleration is used for values <= UINT64_MAX.
//   - Larger values use arbitrary-precision CPU Pollard-Rho.
//   - Every returned factor is verified exactly.
//

// ------------------------------------------------------------
// Work around an nvcc / GCC 13 (also seen on 11/12/14 with some
// CUDA Toolkit 12.x builds) incompatibility: GCC's AMX intrinsic
// headers -- pulled in transitively by <immintrin.h>, which system
// headers such as <random>/<chrono> can drag in -- call compiler
// builtins like __builtin_ia32_ldtilecfg / __builtin_ia32_sttilecfg
// that nvcc's own front-end does not recognise, e.g.:
//
//   amxtileintrin.h(42): error: identifier "__builtin_ia32_ldtilecfg"
//   is undefined
//
// This program never uses AMX tile instructions, so we pre-define
// each AMX header's own include guard *before* anything can include
// it. That makes the preprocessor skip those header bodies entirely,
// so the offending builtins never reach nvcc's parser. This has to
// come before every other #include in the translation unit.
// ------------------------------------------------------------
// Add more "_XXXINTRIN_H_INCLUDED" guards here if a future GCC/CUDA
// combination surfaces the same style of error for a different
// intrinsics header -- the fix is always the same: pre-define that
// header's own guard macro (visible in the error's file path, e.g.
// ".../include/whatever.h") so its body is skipped.
#if defined(__CUDACC__)
#define _AMXTILEINTRIN_H_INCLUDED
#define _AMXINT8INTRIN_H_INCLUDED
#define _AMXBF16INTRIN_H_INCLUDED
#define _AMXFP16INTRIN_H_INCLUDED
#define _AMXCOMPLEXINTRIN_H_INCLUDED
#define _AMXTF32INTRIN_H_INCLUDED
#define _AMXFP8INTRIN_H_INCLUDED
#define _AMXAVX512INTRIN_H_INCLUDED
#define _AMXTRANSPOSEINTRIN_H_INCLUDED
#define _AMXMOVRSINTRIN_H_INCLUDED
#define _AMXMOVRSTRANSPOSEINTRIN_H_INCLUDED
#define _AVX512BF16INTRIN_H_INCLUDED
#define _AVX512BF16VLINTRIN_H_INCLUDED
#define _AVX512FP16INTRIN_H_INCLUDED
#define _AVX512FP16VLINTRIN_H_INCLUDED
#endif

#include <cuda_runtime.h>

#include <boost/multiprecision/cpp_int.hpp>
#ifdef __CUDACC__
#define BOOST_DISABLE_ASSERTS
#define BOOST_NO_CXX11_HDR_ARRAY
#define BOOST_NO_CXX11_HDR_ATOMIC
#define BOOST_NO_CXX11_HDR_TUPLE
#endif

#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>


using boost::multiprecision::cpp_int;


// ============================================================
// CUDA ERROR CHECKING
// ============================================================

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t error__ = (call);                        \
        if (error__ != cudaSuccess) {                         \
            std::ostringstream message__;                     \
            message__                                           \
                << "CUDA error at " << __FILE__                \
                << ":" << __LINE__ << ": "                    \
                << cudaGetErrorString(error__);               \
            throw std::runtime_error(message__.str());        \
        }                                                     \
    } while (false)


// ============================================================
// CONFIGURATION
// ============================================================

constexpr int CUDA_BLOCKS = 256;
constexpr int CUDA_THREADS = 256;
constexpr int CUDA_ROUNDS = 4096;
constexpr int CUDA_RESTARTS = 64;


// ============================================================
// HOST PRIME TABLES
// ============================================================

constexpr std::array<uint64_t, 16> HOST_SMALL_PRIMES = {
    2ULL,
    3ULL,
    5ULL,
    7ULL,
    11ULL,
    13ULL,
    17ULL,
    19ULL,
    23ULL,
    29ULL,
    31ULL,
    37ULL,
    41ULL,
    43ULL,
    47ULL,
    53ULL
};


constexpr std::array<uint64_t, 12>
HOST_MILLER_RABIN_BASES = {
    2ULL,
    3ULL,
    5ULL,
    7ULL,
    11ULL,
    13ULL,
    17ULL,
    19ULL,
    325ULL,
    9375ULL,
    450775ULL,
    1795265022ULL
};


// ============================================================
// DEVICE CONSTANTS
// ============================================================

__constant__ uint64_t DEVICE_SMALL_PRIMES[16] = {
    2ULL,
    3ULL,
    5ULL,
    7ULL,
    11ULL,
    13ULL,
    17ULL,
    19ULL,
    23ULL,
    29ULL,
    31ULL,
    37ULL,
    41ULL,
    43ULL,
    47ULL,
    53ULL
};


__constant__ uint64_t DEVICE_MILLER_RABIN_BASES[12] = {
    2ULL,
    3ULL,
    5ULL,
    7ULL,
    11ULL,
    13ULL,
    17ULL,
    19ULL,
    325ULL,
    9375ULL,
    450775ULL,
    1795265022ULL
};


// ============================================================
// 64-BIT DEVICE ARITHMETIC
// ============================================================

__device__
uint64_t device_add_mod(
    uint64_t a,
    uint64_t b,
    uint64_t n
) {
    if (a >= n - b) {
        return a - (n - b);
    }

    return a + b;
}


__device__
uint64_t device_mul_mod(
    uint64_t a,
    uint64_t b,
    uint64_t n
) {
    uint64_t result = 0ULL;

    a %= n;

    while (b != 0ULL) {
        if (b & 1ULL) {
            result =
                device_add_mod(
                    result,
                    a,
                    n
                );
        }

        b >>= 1ULL;

        if (b != 0ULL) {
            a =
                device_add_mod(
                    a,
                    a,
                    n
                );
        }
    }

    return result;
}


__device__
uint64_t device_pow_mod(
    uint64_t base,
    uint64_t exponent,
    uint64_t n
) {
    uint64_t result = 1ULL % n;

    base %= n;

    while (exponent != 0ULL) {
        if (exponent & 1ULL) {
            result =
                device_mul_mod(
                    result,
                    base,
                    n
                );
        }

        exponent >>= 1ULL;

        if (exponent != 0ULL) {
            base =
                device_mul_mod(
                    base,
                    base,
                    n
                );
        }
    }

    return result;
}


__device__
uint64_t device_gcd(
    uint64_t a,
    uint64_t b
) {
    while (b != 0ULL) {
        uint64_t remainder = a % b;
        a = b;
        b = remainder;
    }

    return a;
}


__device__
uint64_t device_rho_function(
    uint64_t x,
    uint64_t c,
    uint64_t n
) {
    return device_add_mod(
        device_mul_mod(x, x, n),
        c,
        n
    );
}


// ============================================================
// DEVICE RANDOM GENERATOR
// ============================================================

__device__
uint64_t device_splitmix64(
    uint64_t& state
) {
    state += 0x9E3779B97F4A7C15ULL;

    uint64_t z = state;

    z = (
        z ^ (z >> 30)
    ) * 0xBF58476D1CE4E5B9ULL;

    z = (
        z ^ (z >> 27)
    ) * 0x94D049BB133111EBULL;

    return z ^ (z >> 31);
}


__device__
uint64_t device_random_below(
    uint64_t& state,
    uint64_t bound
) {
    return device_splitmix64(state) % bound;
}


// ============================================================
// CUDA POLLARD-RHO KERNEL
// ============================================================

__global__
void pollard_rho_kernel(
    uint64_t n,
    int rounds,
    uint64_t seed,
    unsigned long long* result
) {
    uint64_t thread_id =
        static_cast<uint64_t>(
            blockIdx.x
        )
        * static_cast<uint64_t>(blockDim.x)
        + static_cast<uint64_t>(threadIdx.x);

    uint64_t state =
        seed
        ^ (
            thread_id
            * 0x9E3779B97F4A7C15ULL
        );

    if (n < 4ULL) {
        return;
    }

    if ((n & 1ULL) == 0ULL) {
        atomicCAS(result, 0ULL, 2ULL);
        return;
    }

    if (n % 3ULL == 0ULL) {
        atomicCAS(result, 0ULL, 3ULL);
        return;
    }

    if (n % 5ULL == 0ULL) {
        atomicCAS(result, 0ULL, 5ULL);
        return;
    }

    uint64_t c =
        1ULL
        + device_random_below(
            state,
            n - 1ULL
        );

    uint64_t x =
        2ULL
        + device_random_below(
            state,
            n - 2ULL
        );

    uint64_t y = x;

    for (int iteration = 0; iteration < rounds; ++iteration) {
        if (*result != 0ULL) {
            return;
        }

        x =
            device_rho_function(
                x,
                c,
                n
            );

        y =
            device_rho_function(
                y,
                c,
                n
            );

        y =
            device_rho_function(
                y,
                c,
                n
            );

        uint64_t distance =
            x >= y
            ? x - y
            : y - x;

        uint64_t divisor =
            device_gcd(
                distance,
                n
            );

        if (
            divisor > 1ULL
            && divisor < n
        ) {
            atomicCAS(
                result,
                0ULL,
                static_cast<unsigned long long>(
                    divisor
                )
            );

            return;
        }

        if (divisor == n) {
            return;
        }
    }
}


// ============================================================
// CUDA FACTOR FINDER
// ============================================================

class CudaFactorFinder {
public:
    CudaFactorFinder() {
        CUDA_CHECK(
            cudaMalloc(
                reinterpret_cast<void**>(
                    &device_result_
                ),
                sizeof(unsigned long long)
            )
        );
    }

    ~CudaFactorFinder() {
        if (device_result_ != nullptr) {
            cudaFree(device_result_);
        }
    }

    CudaFactorFinder(
        const CudaFactorFinder&
    ) = delete;

    CudaFactorFinder& operator=(
        const CudaFactorFinder&
    ) = delete;

    uint64_t find_factor(
        uint64_t n
    ) {
        for (
            int restart = 0;
            restart < CUDA_RESTARTS;
            ++restart
        ) {
            unsigned long long zero = 0ULL;

            CUDA_CHECK(
                cudaMemcpy(
                    device_result_,
                    &zero,
                    sizeof(unsigned long long),
                    cudaMemcpyHostToDevice
                )
            );

            uint64_t seed =
                static_cast<uint64_t>(
                    std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count()
                )
                ^ n
                ^ static_cast<uint64_t>(
                    restart
                );

            pollard_rho_kernel<<<
                CUDA_BLOCKS,
                CUDA_THREADS
            >>>(
                n,
                CUDA_ROUNDS,
                seed,
                device_result_
            );

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            unsigned long long result = 0ULL;

            CUDA_CHECK(
                cudaMemcpy(
                    &result,
                    device_result_,
                    sizeof(unsigned long long),
                    cudaMemcpyDeviceToHost
                )
            );

            uint64_t factor =
                static_cast<uint64_t>(
                    result
                );

            if (
                factor > 1ULL
                && factor < n
                && n % factor == 0ULL
            ) {
                return factor;
            }
        }

        return 0ULL;
    }

private:
    unsigned long long* device_result_ = nullptr;
};


// ============================================================
// BIG INTEGER RANDOM SOURCE
// ============================================================

std::mt19937_64 host_rng(
    std::random_device{}()
);


cpp_int random_cpp_int_below(
    const cpp_int& bound
) {
    if (bound <= 1) {
        return cpp_int(1);
    }

    cpp_int result = 0;

    unsigned int bit_count =
        boost::multiprecision::msb(bound) + 1;

    unsigned int limb_count =
        (bit_count + 63U) / 64U;

    do {
        result = 0;

        for (unsigned int i = 0; i < limb_count; ++i) {
            result <<= 64;
            result += host_rng();
        }

        result %= bound;
    }
    while (result == 0);

    return result;
}


// ============================================================
// BIG INTEGER MODULAR ARITHMETIC
// ============================================================

cpp_int gcd_cpp_int(
    cpp_int a,
    cpp_int b
) {
    while (b != 0) {
        cpp_int remainder =
            a % b;

        a = b;
        b = remainder;
    }

    return a;
}


cpp_int power_mod_cpp_int(
    cpp_int base,
    cpp_int exponent,
    const cpp_int& modulus
) {
    cpp_int result = 1;

    base %= modulus;

    while (exponent != 0) {
        if ((exponent & 1) != 0) {
            result =
                (result * base)
                % modulus;
        }

        exponent >>= 1;

        if (exponent != 0) {
            base =
                (base * base)
                % modulus;
        }
    }

    return result;
}


// ============================================================
// BIG INTEGER MILLER-RABIN
// ============================================================

bool is_probable_prime(
    const cpp_int& n
) {
    if (n < 2) {
        return false;
    }

    for (uint64_t prime : HOST_SMALL_PRIMES) {
        cpp_int p = prime;

        if (n == p) {
            return true;
        }

        if (n % p == 0) {
            return false;
        }
    }

    cpp_int d = n - 1;
    unsigned int s = 0;

    while ((d & 1) == 0) {
        d >>= 1;
        ++s;
    }

    for (uint64_t base : HOST_MILLER_RABIN_BASES) {
        cpp_int a = base % n;

        if (a == 0) {
            continue;
        }

        cpp_int x =
            power_mod_cpp_int(
                a,
                d,
                n
            );

        if (x == 1 || x == n - 1) {
            continue;
        }

        bool witness = true;

        for (unsigned int r = 1; r < s; ++r) {
            x =
                (x * x)
                % n;

            if (x == n - 1) {
                witness = false;
                break;
            }
        }

        if (witness) {
            return false;
        }
    }

    return true;
}


// ============================================================
// BIG INTEGER POLLARD-RHO
// ============================================================

cpp_int rho_function(
    const cpp_int& x,
    const cpp_int& c,
    const cpp_int& n
) {
    return (
        (x * x + c)
        % n
    );
}


cpp_int pollard_rho_cpp_int(
    const cpp_int& n,
    int max_iterations = 100000
) {
    if (n % 2 == 0) {
        return 2;
    }

    if (n % 3 == 0) {
        return 3;
    }

    if (n % 5 == 0) {
        return 5;
    }

    if (is_probable_prime(n)) {
        return n;
    }

    for (;;) {
        cpp_int c =
            random_cpp_int_below(n);

        cpp_int x =
            random_cpp_int_below(n);

        cpp_int y = x;

        for (
            int iteration = 0;
            iteration < max_iterations;
            ++iteration
        ) {
            x =
                rho_function(
                    x,
                    c,
                    n
                );

            y =
                rho_function(
                    y,
                    c,
                    n
                );

            y =
                rho_function(
                    y,
                    c,
                    n
                );

            cpp_int difference =
                x >= y
                ? x - y
                : y - x;

            cpp_int divisor =
                gcd_cpp_int(
                    difference,
                    n
                );

            if (
                divisor > 1
                && divisor < n
            ) {
                return divisor;
            }

            if (divisor == n) {
                break;
            }
        }
    }
}


// ============================================================
// COMPLETE FACTORISATION
// ============================================================

void factor_recursive(
    const cpp_int& n,
    std::vector<cpp_int>& factors,
    CudaFactorFinder& cuda_finder
) {
    if (n < 2) {
        return;
    }

    if (is_probable_prime(n)) {
        factors.push_back(n);
        return;
    }

    cpp_int divisor;

    bool fits_u64 =
        n <= std::numeric_limits<uint64_t>::max();

    if (fits_u64) {
        uint64_t n64 =
            n.convert_to<uint64_t>();

        uint64_t factor =
            cuda_finder.find_factor(n64);

        if (factor != 0ULL) {
            divisor = factor;
        }
    }

    if (divisor == 0) {
        divisor =
            pollard_rho_cpp_int(n);
    }

    if (
        divisor <= 1
        || divisor >= n
        || n % divisor != 0
    ) {
        throw std::runtime_error(
            "Invalid Pollard-Rho divisor"
        );
    }

    cpp_int quotient =
        n / divisor;

    factor_recursive(
        divisor,
        factors,
        cuda_finder
    );

    factor_recursive(
        quotient,
        factors,
        cuda_finder
    );
}


std::vector<cpp_int> factor_complete(
    const cpp_int& n,
    CudaFactorFinder& cuda_finder
) {
    std::vector<cpp_int> factors;

    factor_recursive(
        n,
        factors,
        cuda_finder
    );

    std::sort(
        factors.begin(),
        factors.end()
    );

    return factors;
}


// ============================================================
// VERIFICATION
// ============================================================

bool verify_factorisation(
    const cpp_int& n,
    const std::vector<cpp_int>& factors
) {
    if (n < 2) {
        return factors.empty();
    }

    cpp_int product = 1;

    for (const cpp_int& factor : factors) {
        if (!is_probable_prime(factor)) {
            return false;
        }

        product *= factor;
    }

    return product == n;
}


// ============================================================
// FORMATTING
// ============================================================

std::string format_factorisation(
    const std::vector<cpp_int>& factors
) {
    if (factors.empty()) {
        return "1";
    }

    std::ostringstream output;

    std::size_t index = 0;

    while (index < factors.size()) {
        const cpp_int& factor =
            factors[index];

        std::size_t count = 0;

        while (
            index + count < factors.size()
            && factors[index + count] == factor
        ) {
            ++count;
        }

        if (index > 0) {
            output << " * ";
        }

        output << factor;

        if (count > 1) {
            output << "^" << count;
        }

        index += count;
    }

    return output.str();
}


// ============================================================
// SOLVE
// ============================================================

void solve(
    const cpp_int& n,
    CudaFactorFinder& cuda_finder
) {
    std::cout << "\n";
    std::cout << "N = " << n << "\n";

    if (n < 2) {
        std::cout
            << "N must be at least 2.\n";
        return;
    }

    auto start =
        std::chrono::high_resolution_clock::now();

    std::vector<cpp_int> factors =
        factor_complete(
            n,
            cuda_finder
        );

    auto finish =
        std::chrono::high_resolution_clock::now();

    bool verified =
        verify_factorisation(
            n,
            factors
        );

    double elapsed =
        std::chrono::duration<double>(
            finish - start
        ).count();

    std::cout
        << "Prime factors: [";

    for (
        std::size_t index = 0;
        index < factors.size();
        ++index
    ) {
        if (index > 0) {
            std::cout << ", ";
        }

        std::cout << factors[index];
    }

    std::cout << "]\n";

    std::cout
        << "Prime factorisation: "
        << format_factorisation(factors)
        << "\n";

    std::cout
        << "Complete verification: "
        << (
            verified
            ? "true"
            : "false"
        )
        << "\n";

    std::cout
        << "Time: "
        << std::fixed
        << std::setprecision(6)
        << elapsed
        << " s\n";
}


// ============================================================
// CUDA INFORMATION
// ============================================================

void print_cuda_info() {
    int count = 0;

    cudaError_t status =
        cudaGetDeviceCount(&count);

    if (
        status != cudaSuccess
        || count == 0
    ) {
        std::cout
            << "CUDA device unavailable.\n"
            << "Large-integer CPU mode remains available.\n";

        return;
    }

    cudaDeviceProp properties{};

    CUDA_CHECK(
        cudaGetDeviceProperties(
            &properties,
            0
        )
    );

    std::cout
        << "CUDA device: "
        << properties.name
        << "\n";
}


// ============================================================
// MAIN
// ============================================================

int main(
    int argc,
    char** argv
) {
    try {
        print_cuda_info();

        CudaFactorFinder cuda_finder;

        if (argc < 2) {
            std::cout
                << "Usage:\n"
                << "  ./factor_boost INTEGER\n"
                << "  ./factor_boost benchmark\n";

            return 0;
        }

        std::string input =
            argv[1];

        if (input == "benchmark") {
            const std::vector<std::string> examples = {
                "360",
                "8051",
                "18446744073709551617",
                "340282366920938463463374607431768211507"
            };

            for (
                const std::string& value :
                examples
            ) {
                cpp_int n(value);

                solve(
                    n,
                    cuda_finder
                );
            }

            return 0;
        }

        cpp_int n(input);

        solve(
            n,
            cuda_finder
        );
    }
    catch (const std::exception& error) {
        std::cerr
            << "Error: "
            << error.what()
            << "\n";

        return 1;
    }

    return 0;
}
