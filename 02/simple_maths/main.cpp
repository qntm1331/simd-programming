#include "../../include/simd_utils.h"
#include <iostream>
#include <iomanip>
#include <cmath>

/**
 * SIMD 기본 수학 연산
 *
 * 1. 덧셈 (_mm256_add_ps)
 * 2. 뺄셈 (_mm256_sub_ps)
 * 3. 곱셈 (_mm256_mul_ps)
 * 4. 나눗셈 (_mm256_div_ps)
 * 5. 융합 곱셈-덧셈 (_mm256_fmadd_ps)
 * 6. 제곱근 (_mm256_sqrt_ps)
 * 7. 최소/최대 (_mm256_min_ps, _mm256_max_ps)
 * 8. 수평 연산 (_mm256_hadd_ps, _mm256_hsub_ps)
 *
 * 각 연산마다 SIMD와 스칼라 구현의 성능을 비교
 */

int main() {
    std::cout << "=== SIMD 수학 연산 ===" << std::endl;
    std::cout << std::endl;

    // 테스트 데이터 초기화
    float data1[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float data2[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    // 데이터를 SIMD 벡터로 로드
    __m256 vector1 = _mm256_loadu_ps(data1);
    __m256 vector2 = _mm256_loadu_ps(data2);

    // ========================================================================
    // 1. 덧셈
    // ========================================================================
    std::cout << "1. 덧셈 (_mm256_add_ps)" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "두 벡터의 대응하는 요소들을 더합니다." << std::endl;
    std::cout << std::endl;

    print_m256(vector1, "벡터 1");
    print_m256(vector2, "벡터 2");

    // 덧셈 수행 (8개 요소를 동시에)
    __m256 add_result = _mm256_add_ps(vector1, vector2);
    print_m256(add_result, "덧셈 결과 (벡터 1 + 벡터 2)");

    // 성능 비교: 스칼라 vs. SIMD
    auto scalar_add = [&]() {
        float result[8];
        for (int i = 0; i < 8; i++) {
            result[i] = data1[i] + data2[i];  // 8번 반복
        }
    };

    auto simd_add = [&]() {
        __m256 result = _mm256_add_ps(vector1, vector2);  // 1번에 8개 처리
    };

    benchmark_comparison("덧셈", scalar_add, simd_add);
    std::cout << std::endl;

    // ========================================================================
    // 2. 뺄셈
    // ========================================================================
    std::cout << "2. 뺄셈 (_mm256_sub_ps)" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "두 벡터의 대응하는 요소들을 뺍니다." << std::endl;
    std::cout << std::endl;

    // 뺄셈 수행
    __m256 sub_result = _mm256_sub_ps(vector1, vector2);
    print_m256(sub_result, "뺄셈 결과 (벡터 1 - 벡터 2)");

    // 성능 비교: 스칼라 vs. SIMD
    auto scalar_sub = [&]() {
        float result[8];
        for (int i = 0; i < 8; i++) {
            result[i] = data1[i] - data2[i];
        }
    };

    auto simd_sub = [&]() {
        __m256 result = _mm256_sub_ps(vector1, vector2);
    };

    benchmark_comparison("뺄셈", scalar_sub, simd_sub);
    std::cout << std::endl;

    // ========================================================================
    // 3. 곱셈
    // ========================================================================
    std::cout << "3. 곱셈 (_mm256_mul_ps)" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "두 벡터의 대응하는 요소들을 곱합니다." << std::endl;
    std::cout << std::endl;

    // 곱셈 수행
    __m256 mul_result = _mm256_mul_ps(vector1, vector2);
    print_m256(mul_result, "곱셈 결과 (벡터 1 * 벡터 2)");

    // 성능 비교: 스칼라 vs. SIMD
    auto scalar_mul = [&]() {
        float result[8];
        for (int i = 0; i < 8; i++) {
            result[i] = data1[i] * data2[i];
        }
    };

    auto simd_mul = [&]() {
        __m256 result = _mm256_mul_ps(vector1, vector2);
    };

    benchmark_comparison("곱셈", scalar_mul, simd_mul);
    std::cout << std::endl;

    // ========================================================================
    // 4. 나눗셈
    // ========================================================================
    std::cout << "4. 나눗셈 (_mm256_div_ps)" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "두 벡터의 대응하는 요소들을 나눕니다." << std::endl;
    std::cout << std::endl;

    // 나눗셈 수행
    __m256 div_result = _mm256_div_ps(vector1, vector2);
    print_m256(div_result, "나눗셈 결과 (벡터 1 / 벡터 2)");

    // 성능 비교: 스칼라 vs. SIMD
    auto scalar_div = [&]() {
        float result[8];
        for (int i = 0; i < 8; i++) {
            result[i] = data1[i] / data2[i];
        }
    };

    auto simd_div = [&]() {
        __m256 result = _mm256_div_ps(vector1, vector2);
    };

    benchmark_comparison("나눗셈", scalar_div, simd_div);
    std::cout << std::endl;

    // ========================================================================
    // 5. 융합 곱셈-덧셈 (FMA)
    // ========================================================================
    std::cout << "5. 융합 곱셈-덧셈 (_mm256_fmadd_ps)" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "융합 곱셈-덧셈 연산을 수행: a*b + c" << std::endl;
    std::cout << "별도의 곱셈과 덧셈보다 더 정확하고 빠릅니다." << std::endl;
    std::cout << std::endl;

    // FMA용 세 번째 벡터 생성
    __m256 vector3 = _mm256_set1_ps(2.0f);
    print_m256(vector3, "벡터 3");

    // FMA 수행: vector1 * vector2 + vector3
    // 1개 명령어로 곱셈과 덧셈을 동시에 수행 (더 빠르고 정확)
    __m256 fma_result = _mm256_fmadd_ps(vector1, vector2, vector3);
    print_m256(fma_result, "FMA 결과 (벡터 1 * 벡터 2 + 벡터 3)");

    // 성능 비교: 스칼라 vs. SIMD
    auto scalar_fma = [&]() {
        float result[8];
        for (int i = 0; i < 8; i++) {
            result[i] = data1[i] * data2[i] + 2.0f;  // 2개 연산
        }
    };

    auto simd_fma = [&]() {
        __m256 result = _mm256_fmadd_ps(vector1, vector2, vector3);  // 1개 명령어
    };

    benchmark_comparison("융합 곱셈-덧셈", scalar_fma, simd_fma);
    std::cout << std::endl;

    // ========================================================================
    // 6. 제곱근
    // ========================================================================
    std::cout << "6. 제곱근 (_mm256_sqrt_ps)" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "벡터의 각 요소에 대한 제곱근을 계산합니다." << std::endl;
    std::cout << std::endl;

    // 양수 값들로 벡터 생성
    __m256 pos_vector = _mm256_set_ps(64.0f, 49.0f, 36.0f, 25.0f, 16.0f, 9.0f, 4.0f, 1.0f);
    print_m256(pos_vector, "입력 벡터");

    // 제곱근 계산 (8개 동시에)
    __m256 sqrt_result = _mm256_sqrt_ps(pos_vector);
    print_m256(sqrt_result, "제곱근 결과");

    // 성능 비교: 스칼라 vs. SIMD
    auto scalar_sqrt = [&]() {
        float result[8];
        union {
            __m256 v;
            float a[8];
        } u;
        u.v = pos_vector;
        for (int i = 0; i < 8; i++) {
            result[i] = std::sqrt(u.a[i]);  // 8번의 sqrt 호출
        }
    };

    auto simd_sqrt = [&]() {
        __m256 result = _mm256_sqrt_ps(pos_vector);  // 1번에 8개 처리
    };

    benchmark_comparison("제곱근", scalar_sqrt, simd_sqrt);
    std::cout << std::endl;

    // ========================================================================
    // 7. 최소/최대 연산
    // ========================================================================
    std::cout << "7. 최소/최대 연산 (_mm256_min_ps, _mm256_max_ps)" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "대응하는 요소들의 최소값 또는 최대값을 계산합니다." << std::endl;
    std::cout << std::endl;

    print_m256(vector1, "벡터 1");
    print_m256(vector2, "벡터 2");

    // 최소값과 최대값 계산
    __m256 min_result = _mm256_min_ps(vector1, vector2);
    __m256 max_result = _mm256_max_ps(vector1, vector2);

    print_m256(min_result, "최소값 결과");
    print_m256(max_result, "최대값 결과");

    // 성능 비교: 스칼라 vs. SIMD (최소값)
    auto scalar_min = [&]() {
        float result[8];
        for (int i = 0; i < 8; i++) {
            result[i] = std::min(data1[i], data2[i]);
        }
    };

    auto simd_min = [&]() {
        __m256 result = _mm256_min_ps(vector1, vector2);
    };

    benchmark_comparison("최소값", scalar_min, simd_min);
    std::cout << std::endl;

    // ========================================================================
    // 8. 수평 연산 (Horizontal Operations)
    // ========================================================================
    std::cout << "8. 수평 연산 (_mm256_hadd_ps, _mm256_hsub_ps)" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "인접한 요소들의 수평 덧셈 또는 뺄셈을 수행합니다." << std::endl;
    std::cout << std::endl;

    // 테스트 벡터 생성
    __m256 hadd_vec1 = _mm256_set_ps(8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f);
    __m256 hadd_vec2 = _mm256_set_ps(16.0f, 15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f);

    print_m256(hadd_vec1, "벡터 A");
    print_m256(hadd_vec2, "벡터 B");

    // 수평 덧셈 수행
    // 인접한 쌍을 더함: (a0+a1, a2+a3, b0+b1, b2+b3, a4+a5, a6+a7, b4+b5, b6+b7)
    __m256 hadd_result = _mm256_hadd_ps(hadd_vec1, hadd_vec2);
    print_m256(hadd_result, "수평 덧셈 결과");

    // 수평 뺄셈 수행
    // 인접한 쌍을 뺌: (a0-a1, a2-a3, b0-b1, b2-b3, a4-a5, a6-a7, b4-b5, b6-b7)
    __m256 hsub_result = _mm256_hsub_ps(hadd_vec1, hadd_vec2);
    print_m256(hsub_result, "수평 뺄셈 결과");

    // 참고: 수평 연산은 일반적으로 수직 연산보다 느립니다
    // 내적(dot product)이나 행렬 연산 같은 특정 알고리즘에 유용합니다

    return 0;
}
