#include "../../include/simd_utils.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>

/**
 * SIMD를 사용한 벡터 내적 구현
 *
 * SIMD로 내적을 계산하는 방법
 * 1. 스칼라 구현 (기준선)
 * 2. x, y, z 컴포넌트를 별도 벡터로 사용하는 기본 SIMD 구현
 * 3. Structure of Arrays (SoA) 레이아웃을 사용하는 SIMD 구현
 * 4. 수평 덧셈을 사용하는 SIMD 구현
 * 5. 대용량 배열을 위한 SIMD 구현 (배치 처리)
 *
 * 내적은 다음 분야에서 기본 연산입니다:
 * - 컴퓨터 그래픽스 (조명 계산, 투영)
 * - 머신러닝 (신경망, 유사도 측정)
 * - 물리 시뮬레이션 (힘 계산)
 */

// 3D 벡터 구조체 (Array of Structures 레이아웃)
struct Vec3 {
    float x, y, z;

    Vec3(float x = 0.0f, float y = 0.0f, float z = 0.0f) : x(x), y(y), z(z) {}

    // 스칼라 내적
    // dot(a, b) = a.x*b.x + a.y*b.y + a.z*b.z
    float dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
};

// Structure of Arrays 레이아웃 (SIMD 성능 향상)
// AoS: [x1,y1,z1][x2,y2,z2][x3,y3,z3]...  ← 메모리에서 떨어져 있음
// SoA: [x1,x2,x3...][y1,y2,y3...][z1,z2,z3...]  ← 같은 컴포넌트가 연속
struct Vec3Array {
    std::vector<float> x;  // 모든 x 컴포넌트
    std::vector<float> y;  // 모든 y 컴포넌트
    std::vector<float> z;  // 모든 z 컴포넌트

    Vec3Array(size_t size) : x(size), y(size), z(size) {}

    void set(size_t index, float x_val, float y_val, float z_val) {
        x[index] = x_val;
        y[index] = y_val;
        z[index] = z_val;
    }

    void set(size_t index, const Vec3& vec) {
        x[index] = vec.x;
        y[index] = vec.y;
        z[index] = vec.z;
    }
};

// 랜덤 3D 벡터 생성
std::vector<Vec3> generateRandomVectors(size_t count) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<Vec3> vectors;
    vectors.reserve(count);

    for (size_t i = 0; i < count; i++) {
        vectors.emplace_back(dist(gen), dist(gen), dist(gen));
    }

    return vectors;
}

// Array of Structures를 Structure of Arrays로 변환
Vec3Array convertToSoA(const std::vector<Vec3>& vectors) {
    Vec3Array result(vectors.size());

    for (size_t i = 0; i < vectors.size(); i++) {
        result.set(i, vectors[i]);
    }

    return result;
}

// ============================================================================
// 1. 스칼라 내적 구현
// ============================================================================
float scalarDotProduct(const std::vector<Vec3>& vectors1, const std::vector<Vec3>& vectors2) {
    float sum = 0.0f;
    for (size_t i = 0; i < vectors1.size(); i++) {
        sum += vectors1[i].dot(vectors2[i]);  // 각 벡터 쌍의 내적을 계산하여 합산
    }
    return sum;
}

// ============================================================================
// 2. 기본 SIMD 내적 구현 (한 번에 8개 벡터)
// ============================================================================
__m256 simdDotProduct8(const std::vector<Vec3>& vectors1, const std::vector<Vec3>& vectors2) {
    // x, y, z 컴포넌트를 별도의 SIMD 레지스터로 로드
    float x1[8], y1[8], z1[8];
    float x2[8], y2[8], z2[8];

    // AoS에서 SoA로 수동 변환 (메모리 재배치)
    for (int i = 0; i < 8; i++) {
        x1[i] = vectors1[i].x;
        y1[i] = vectors1[i].y;
        z1[i] = vectors1[i].z;
        x2[i] = vectors2[i].x;
        y2[i] = vectors2[i].y;
        z2[i] = vectors2[i].z;
    }

    // 데이터를 SIMD 레지스터로 로드
    __m256 vx1 = _mm256_loadu_ps(x1);
    __m256 vy1 = _mm256_loadu_ps(y1);
    __m256 vz1 = _mm256_loadu_ps(z1);
    __m256 vx2 = _mm256_loadu_ps(x2);
    __m256 vy2 = _mm256_loadu_ps(y2);
    __m256 vz2 = _mm256_loadu_ps(z2);

    // FMA를 사용하여 내적 계산
    // dot = x1*x2 + y1*y2 + z1*z2
    __m256 result = _mm256_mul_ps(vx1, vx2);                  // x1*x2
    result = _mm256_fmadd_ps(vy1, vy2, result);               // result + y1*y2
    result = _mm256_fmadd_ps(vz1, vz2, result);               // result + z1*z2

    return result;  // 8개의 내적 결과를 한 번에 반환
}

// ============================================================================
// 3. Structure of Arrays 레이아웃을 사용한 SIMD 내적
// ============================================================================
__m256 simdDotProductSoA8(const Vec3Array& vectors1, const Vec3Array& vectors2, size_t offset) {
    // SoA 구조에서 직접 SIMD 레지스터로 로드 (메모리 재배치 불필요!)
    __m256 vx1 = _mm256_loadu_ps(&vectors1.x[offset]);
    __m256 vy1 = _mm256_loadu_ps(&vectors1.y[offset]);
    __m256 vz1 = _mm256_loadu_ps(&vectors1.z[offset]);
    __m256 vx2 = _mm256_loadu_ps(&vectors2.x[offset]);
    __m256 vy2 = _mm256_loadu_ps(&vectors2.y[offset]);
    __m256 vz2 = _mm256_loadu_ps(&vectors2.z[offset]);

    // FMA를 사용하여 내적 계산
    __m256 result = _mm256_mul_ps(vx1, vx2);
    result = _mm256_fmadd_ps(vy1, vy2, result);
    result = _mm256_fmadd_ps(vz1, vz2, result);

    return result;
}

// ============================================================================
// 4. 수평 덧셈을 사용한 SIMD 내적 (단일 내적)
// ============================================================================
float simdDotProductSingle(const Vec3& v1, const Vec3& v2) {
    // 벡터 컴포넌트를 SIMD 레지스터로 로드
    __m128 vec1 = _mm_setr_ps(v1.x, v1.y, v1.z, 0.0f);
    __m128 vec2 = _mm_setr_ps(v2.x, v2.y, v2.z, 0.0f);

    // 컴포넌트별 곱셈
    __m128 mul = _mm_mul_ps(vec1, vec2);  // [x1*x2, y1*y2, z1*z2, 0]

    // 수평 덧셈으로 컴포넌트 합산
    // 첫 번째 hadd: (x+y, z+0, x+y, z+0)
    __m128 hadd1 = _mm_hadd_ps(mul, mul);
    // 두 번째 hadd: (x+y+z+0, x+y+z+0, x+y+z+0, x+y+z+0)
    __m128 hadd2 = _mm_hadd_ps(hadd1, hadd1);

    // 결과 추출 (첫 번째 요소)
    return _mm_cvtss_f32(hadd2);
}

// ============================================================================
// 5. 대용량 배열을 위한 SIMD 내적
// ============================================================================
float simdDotProductLarge(const Vec3Array& vectors1, const Vec3Array& vectors2) {
    size_t size = vectors1.x.size();
    size_t blocks = size / 8;       // 8개씩 처리할 블록 수
    size_t remainder = size % 8;    // 남은 개수

    // 한 번에 8개 벡터 처리
    __m256 sum = _mm256_setzero_ps();
    for (size_t i = 0; i < blocks; i++) {
        __m256 dot8 = simdDotProductSoA8(vectors1, vectors2, i * 8);
        sum = _mm256_add_ps(sum, dot8);  // 8개의 내적 결과를 누적
    }

    // 8개 내적의 수평 합계
    float result_array[8];
    _mm256_storeu_ps(result_array, sum);
    float total = 0.0f;
    for (int i = 0; i < 8; i++) {
        total += result_array[i];
    }

    // 남은 벡터 처리 (8개 미만)
    for (size_t i = blocks * 8; i < size; i++) {
        Vec3 v1(vectors1.x[i], vectors1.y[i], vectors1.z[i]);
        Vec3 v2(vectors2.x[i], vectors2.y[i], vectors2.z[i]);
        total += v1.dot(v2);
    }

    return total;
}

int main() {
    std::cout << "=== SIMD 내적 구현 ===" << std::endl;
    std::cout << std::endl;

    // 랜덤 테스트 벡터 생성
    const size_t NUM_VECTORS = 1024;
    std::vector<Vec3> vectors1 = generateRandomVectors(NUM_VECTORS);
    std::vector<Vec3> vectors2 = generateRandomVectors(NUM_VECTORS);

    // 더 효율적인 SIMD 처리를 위해 Structure of Arrays로 변환
    Vec3Array soa_vectors1 = convertToSoA(vectors1);
    Vec3Array soa_vectors2 = convertToSoA(vectors2);

    // ========================================================================
    // 1. 기본 내적 비교
    // ========================================================================
    std::cout << "1. 기본 내적 (8개 벡터)" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "8개 벡터에 대한 스칼라 vs. SIMD 구현 비교" << std::endl;
    std::cout << std::endl;

    // 스칼라 방식으로 내적 계산
    float scalar_results[8];
    for (int i = 0; i < 8; i++) {
        scalar_results[i] = vectors1[i].dot(vectors2[i]);
    }

    // SIMD로 내적 계산 (8개 동시에)
    __m256 simd_result = simdDotProduct8(vectors1, vectors2);
    float simd_results[8];
    _mm256_storeu_ps(simd_results, simd_result);

    // 결과 출력 및 비교
    std::cout << "스칼라 결과: [";
    for (int i = 0; i < 7; i++) {
        std::cout << scalar_results[i] << ", ";
    }
    std::cout << scalar_results[7] << "]" << std::endl;

    std::cout << "SIMD 결과:   [";
    for (int i = 0; i < 7; i++) {
        std::cout << simd_results[i] << ", ";
    }
    std::cout << simd_results[7] << "]" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // 2. 성능 비교
    // ========================================================================
    std::cout << "2. 성능 비교" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "다양한 내적 구현의 성능 비교" << std::endl;
    std::cout << std::endl;

    // 스칼라 구현 벤치마크
    auto scalar_benchmark = [&]() {
        volatile float result = scalarDotProduct(vectors1, vectors2);
    };

    // AoS 레이아웃의 SIMD 구현 벤치마크
    auto simd_aos_benchmark = [&]() {
        float total = 0.0f;
        for (size_t i = 0; i < NUM_VECTORS; i += 8) {
            size_t remaining = std::min(size_t(8), NUM_VECTORS - i);
            if (remaining < 8) break;  // 간단히 하기 위해 불완전한 블록은 건너뜀

            std::vector<Vec3> block1(vectors1.begin() + i, vectors1.begin() + i + 8);
            std::vector<Vec3> block2(vectors2.begin() + i, vectors2.begin() + i + 8);

            __m256 result = simdDotProduct8(block1, block2);
            float results[8];
            _mm256_storeu_ps(results, result);

            for (int j = 0; j < 8; j++) {
                total += results[j];
            }
        }
    };

    // SoA 레이아웃의 SIMD 구현 벤치마크
    auto simd_soa_benchmark = [&]() {
        volatile float result = simdDotProductLarge(soa_vectors1, soa_vectors2);
    };

    // 벤치마크 실행
    benchmark_comparison("내적 (1024 벡터)", scalar_benchmark, simd_soa_benchmark);
    std::cout << std::endl;

    // ========================================================================
    // 3. Structure of Arrays vs Array of Structures
    // ========================================================================
    std::cout << "3. Structure of Arrays vs Array of Structures" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "SIMD 처리를 위한 AoS vs SoA 메모리 레이아웃 비교" << std::endl;
    std::cout << std::endl;

    benchmark_comparison("AoS vs SoA", simd_aos_benchmark, simd_soa_benchmark);
    std::cout << std::endl;

    // ========================================================================
    // 4. 단일 벡터 내적
    // ========================================================================
    std::cout << "4. 단일 벡터 내적" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "수평 덧셈을 사용한 단일 내적 SIMD 구현" << std::endl;
    std::cout << std::endl;

    Vec3 v1(0.5f, -0.3f, 0.8f);
    Vec3 v2(0.2f, 0.7f, -0.4f);

    float scalar_dot = v1.dot(v2);
    float simd_dot = simdDotProductSingle(v1, v2);

    std::cout << "벡터 1: (" << v1.x << ", " << v1.y << ", " << v1.z << ")" << std::endl;
    std::cout << "벡터 2: (" << v2.x << ", " << v2.y << ", " << v2.z << ")" << std::endl;
    std::cout << "스칼라 내적: " << scalar_dot << std::endl;
    std::cout << "SIMD 내적:   " << simd_dot << std::endl;
    std::cout << std::endl;

    // 단일 벡터 내적 벤치마크
    auto scalar_single_benchmark = [&]() {
        for (int i = 0; i < 1000; i++) {
            volatile float result = v1.dot(v2);
        }
    };

    auto simd_single_benchmark = [&]() {
        for (int i = 0; i < 1000; i++) {
            volatile float result = simdDotProductSingle(v1, v2);
        }
    };

    benchmark_comparison("단일 내적 (1000회 반복)", scalar_single_benchmark, simd_single_benchmark);

    return 0;
}
