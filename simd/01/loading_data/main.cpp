#include "../../include/simd_utils.h"
#include <iostream>
#include <iomanip>
#include <memory>

/**
 * SIMD 데이터 로딩과 저장
 *
 * 이 예제는 SIMD 벡터로 데이터를 로드하는 여러 방법을 보여줍니다:
 * 1. 정렬된 로드 (_mm256_load_ps) - 32바이트 정렬 메모리 필요
 * 2. 정렬되지 않은 로드 (_mm256_loadu_ps) - 모든 메모리 주소에서 작동
 * 3. 마스크 로드 (_mm256_maskload_ps) - 마스크를 기반으로 선택적 로드
 * 4. 스트림 로드 (_mm256_stream_load_si256) - 캐시를 우회하는 논-템포럴 로드
 *
 * SIMD 데이터를 저장하는 방법
 * 1. 정렬된 저장 (_mm256_store_ps) - 32바이트 정렬 메모리 필요
 * 2. 정렬되지 않은 저장 (_mm256_storeu_ps) - 모든 메모리 주소에서 작동
 * 3. 마스크 저장 (_mm256_maskstore_ps) - 마스크를 기반으로 선택적 저장
 * 4. 스트림 저장 (_mm256_stream_ps) - 캐시를 우회하는 논-템포럴 저장
 *
 */

const int ARRAY_SIZE = 8;
const int TEST_ITERATIONS = 10000000;  // 1천만 번 반복 (성능 측정용)

int main() {
    std::cout << "=== SIMD 데이터 로딩과 저장 ===" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // 1. 정렬된 vs. 정렬되지 않은 로드
    // ========================================================================
    std::cout << "1. 정렬된 vs. 정렬되지 않은 로드" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "정렬된 메모리와 정렬되지 않은 메모리 접근 비교" << std::endl;
    std::cout << std::endl;

    // 정렬된 메모리와 정렬되지 않은 메모리 할당
    float* aligned_data = aligned_alloc<float>(ARRAY_SIZE, 32);  // AVX용 32바이트 정렬
    float* unaligned_data = new float[ARRAY_SIZE + 1];           // +1로 정렬 안 된 포인터 생성 가능
    float* unaligned_ptr = unaligned_data + 1;                   // 1칸 오프셋으로 정렬 깨기

    // 데이터 초기화
    for (int i = 0; i < ARRAY_SIZE; i++) {
        aligned_data[i] = static_cast<float>(i + 1);
        unaligned_ptr[i] = static_cast<float>(i + 1);
    }

    // 정렬된 로드 시연
    // 32바이트 정렬된 메모리에서만 사용 가능 (더 빠름)
    __m256 aligned_vec = _mm256_load_ps(aligned_data);
    print_m256(aligned_vec, "정렬된 로드 결과");

    // 정렬되지 않은 로드 시연
    // 모든 주소에서 작동하지만 약간 느림
    __m256 unaligned_vec = _mm256_loadu_ps(unaligned_ptr);
    print_m256(unaligned_vec, "정렬되지 않은 로드 결과");

    // 성능 비교
    Timer timer("정렬된 vs. 정렬되지 않은 로드 성능");

    // 정렬된 로드 벤치마크
    auto aligned_load = [&]() {
        __m256 result;
        for (int i = 0; i < TEST_ITERATIONS; i++) {
            result = _mm256_load_ps(aligned_data);  // 정렬된 로드
        }
        return result;
    };

    // 정렬되지 않은 로드 벤치마크
    auto unaligned_load = [&]() {
        __m256 result;
        for (int i = 0; i < TEST_ITERATIONS; i++) {
            result = _mm256_loadu_ps(unaligned_ptr);  // 정렬 안 된 로드
        }
        return result;
    };

    benchmark_comparison("로드 연산", aligned_load, unaligned_load, 10);
    std::cout << std::endl;

    // ========================================================================
    // 2. 마스크 로드
    // ========================================================================
    std::cout << "2. 마스크 로드" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "마스크를 기반으로 선택적으로 요소 로드하기" << std::endl;
    std::cout << std::endl;

    // 요소 0, 2, 4, 6만 로드하는 마스크 생성
    // -1 (모든 비트 1) = 로드, 0 = 로드 안 함
    __m256i mask = _mm256_set_epi32(0, -1, 0, -1, 0, -1, 0, -1);

    // 마스크 로드 수행 (마스크로 선택되지 않은 요소는 0이 됨)
    __m256 masked_vec = _mm256_maskload_ps(aligned_data, mask);
    print_m256(masked_vec, "마스크 로드 결과 (짝수 인덱스만)");
    std::cout << std::endl;

    // ========================================================================
    // 3. 정렬된 vs. 정렬되지 않은 저장
    // ========================================================================
    std::cout << "3. 정렬된 vs. 정렬되지 않은 저장" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "정렬된 저장과 정렬되지 않은 저장 연산 비교" << std::endl;
    std::cout << std::endl;

    // 테스트용 벡터 생성
    __m256 test_vec = _mm256_set_ps(16.0f, 14.0f, 12.0f, 10.0f, 8.0f, 6.0f, 4.0f, 2.0f);

    // 정렬된 저장 수행
    _mm256_store_ps(aligned_data, test_vec);

    std::cout << "정렬된 저장 결과: [";
    for (int i = 0; i < ARRAY_SIZE - 1; i++) {
        std::cout << aligned_data[i] << ", ";
    }
    std::cout << aligned_data[ARRAY_SIZE - 1] << "]" << std::endl;

    // 정렬되지 않은 저장 수행
    _mm256_storeu_ps(unaligned_ptr, test_vec);

    std::cout << "정렬되지 않은 저장 결과: [";
    for (int i = 0; i < ARRAY_SIZE - 1; i++) {
        std::cout << unaligned_ptr[i] << ", ";
    }
    std::cout << unaligned_ptr[ARRAY_SIZE - 1] << "]" << std::endl;

    // 성능 비교
    Timer timer2("정렬된 vs. 정렬되지 않은 저장 성능");

    // 정렬된 저장 벤치마크
    auto aligned_store = [&]() {
        for (int i = 0; i < TEST_ITERATIONS; i++) {
            _mm256_store_ps(aligned_data, test_vec);  // 정렬된 저장
        }
    };

    // 정렬되지 않은 저장 벤치마크
    auto unaligned_store = [&]() {
        for (int i = 0; i < TEST_ITERATIONS; i++) {
            _mm256_storeu_ps(unaligned_ptr, test_vec);  // 정렬 안 된 저장
        }
    };

    benchmark_comparison("저장 연산", aligned_store, unaligned_store, 10);
    std::cout << std::endl;

    // ========================================================================
    // 4. 마스크 저장
    // ========================================================================
    std::cout << "4. 마스크 저장" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "마스크를 기반으로 선택적으로 요소 저장하기" << std::endl;
    std::cout << std::endl;

    // 정렬된 데이터 초기화
    for (int i = 0; i < ARRAY_SIZE; i++) {
        aligned_data[i] = 0.0f;
    }

    // 요소 1, 3, 5, 7만 저장하는 마스크 생성
    __m256i mask2 = _mm256_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0);

    // 마스크 저장 수행 (마스크가 0인 위치는 변경 안 됨)
    _mm256_maskstore_ps(aligned_data, mask2, test_vec);

    std::cout << "마스크 저장 결과 (홀수 인덱스만): [";
    for (int i = 0; i < ARRAY_SIZE - 1; i++) {
        std::cout << aligned_data[i] << ", ";
    }
    std::cout << aligned_data[ARRAY_SIZE - 1] << "]" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // 5. 스트림 로드/저장 (논-템포럴)
    // ========================================================================
    std::cout << "5. 스트림 로드/저장 (논-템포럴)" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "캐시를 우회하는 논-템포럴 로드와 저장 사용" << std::endl;
    std::cout << "곧 재사용되지 않을 대용량 데이터에 유용합니다." << std::endl;
    std::cout << std::endl;

    // 스트리밍 연산을 보여줄 대용량 배열 할당
    const int LARGE_SIZE = 1024;
    float* large_array = aligned_alloc<float>(LARGE_SIZE, 32);

    // 배열 초기화
    for (int i = 0; i < LARGE_SIZE; i++) {
        large_array[i] = static_cast<float>(i);
    }

    // 스트림 로드와 저장 수행
    for (int i = 0; i < LARGE_SIZE; i += 8) {
        // 스트림 로드 (캐스팅이 필요한 _mm256_stream_load_si256 사용)
        __m256 loaded = _mm256_loadu_ps(&large_array[i]);

        // 데이터 처리 (간단하게 2를 곱함)
        __m256 processed = _mm256_mul_ps(loaded, _mm256_set1_ps(2.0f));

        // 스트림 저장 (캐시를 우회하는 논-템포럴 저장)
        // 대용량 데이터를 한 번만 쓸 때 캐시 오염 방지
        _mm256_stream_ps(&large_array[i], processed);
    }

    // 모든 스트리밍 저장이 완료되도록 보장 (메모리 펜스)
    _mm_sfence();

    std::cout << "스트림 저장 결과 (처음 16개 요소): [";
    for (int i = 0; i < 15; i++) {
        std::cout << large_array[i] << ", ";
    }
    std::cout << large_array[15] << "]" << std::endl;

    free(aligned_data);
    delete[] unaligned_data;
    free(large_array);

    return 0;
}
