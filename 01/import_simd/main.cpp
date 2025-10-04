#include <immintrin.h>
#include <iostream>
#include <iomanip>

// Helper function to print __m256
void print_m256(const __m256& v, const char* label) {
    float result[8];
    _mm256_storeu_ps(result, v);
    std::cout << label << ": ";
    for (int i = 7; i >= 0; i--) {
        std::cout << std::setw(8) << result[i] << " ";
    }
    std::cout << std::endl;
}

// Helper function to print __m256i
void print_m256i(const __m256i& v, const char* label) {
    int result[8];
    _mm256_storeu_si256((__m256i*)result, v);
    std::cout << label << ": ";
    for (int i = 7; i >= 0; i--) {
        std::cout << std::setw(8) << result[i] << " ";
    }
    std::cout << std::endl;
}

// Helper function to print __m256d
void print_m256d(const __m256d& v, const char* label) {
    double result[4];
    _mm256_storeu_pd(result, v);
    std::cout << label << ": ";
    for (int i = 3; i >= 0; i--) {
        std::cout << std::setw(12) << result[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "=== AVX2 SIMD Examples ===" << std::endl << std::endl;
    
    // Example 1: Float operations (8 floats at once)
    std::cout << "--- Float Operations (8-wide) ---" << std::endl;
    __m256 a = _mm256_set_ps(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
    __m256 b = _mm256_set_ps(8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f);
    
    __m256 sum = _mm256_add_ps(a, b);
    __m256 mul = _mm256_mul_ps(a, b);
    __m256 sub = _mm256_sub_ps(a, b);
    
    print_m256(a, "A        ");
    print_m256(b, "B        ");
    print_m256(sum, "A + B    ");
    print_m256(mul, "A * B    ");
    print_m256(sub, "A - B    ");
    
    std::cout << std::endl;
    
    // Example 2: Integer operations (8 integers at once)
    std::cout << "--- Integer Operations (8-wide) ---" << std::endl;
    __m256i int_a = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    __m256i int_b = _mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1);
    __m256i int_sum = _mm256_add_epi32(int_a, int_b);
    __m256i int_mul = _mm256_mullo_epi32(int_a, int_b);
    
    print_m256i(int_a, "A        ");
    print_m256i(int_b, "B        ");
    print_m256i(int_sum, "A + B    ");
    print_m256i(int_mul, "A * B    ");
    
    std::cout << std::endl;
    
    // Example 3: Double operations (4 doubles at once)
    std::cout << "--- Double Operations (4-wide) ---" << std::endl;
    __m256d double_a = _mm256_set_pd(1.0, 2.0, 3.0, 4.0);
    __m256d double_b = _mm256_set_pd(4.0, 3.0, 2.0, 1.0);
    __m256d double_sum = _mm256_add_pd(double_a, double_b);
    __m256d double_mul = _mm256_mul_pd(double_a, double_b);
    
    print_m256d(double_a, "A        ");
    print_m256d(double_b, "B        ");
    print_m256d(double_sum, "A + B    ");
    print_m256d(double_mul, "A * B    ");
    
    std::cout << std::endl;
    
    // Example 4: Practical use case - sum array
    std::cout << "--- Practical Example: Sum Array ---" << std::endl;
    float numbers[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    
    // SIMD version (process 8 at a time)
    __m256 vec1 = _mm256_loadu_ps(&numbers[0]);
    __m256 vec2 = _mm256_loadu_ps(&numbers[8]);
    __m256 total = _mm256_add_ps(vec1, vec2);
    
    // Horizontal sum
    __m128 high = _mm256_extractf128_ps(total, 1);
    __m128 low = _mm256_castps256_ps128(total);
    __m128 sum128 = _mm_add_ps(high, low);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    
    float final_sum;
    _mm_store_ss(&final_sum, sum128);
    
    std::cout << "Sum of array (1-16): " << final_sum << std::endl;
    
    // Verify with scalar
    float scalar_sum = 0;
    for (int i = 0; i < 16; i++) {
        scalar_sum += numbers[i];
    }
    std::cout << "Scalar verification: " << scalar_sum << std::endl;
    
    return 0;
}
