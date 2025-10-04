#include "../../include/simd_utils.h"
#include <iostream>
#include <iomanip>

/**
 * SIMD 데이터 접근 기법들
 *
 * SIMD 벡터의 데이터에 접근하고 조작하는 방법
 * 1. 포인터 변환 사용 (reinterpret_cast)
 * 2. Union을 사용하여 SIMD 타입과 배열 간 alias 생성
 * 3. _mm256_store_*와 _mm256_load_* 함수 사용
 * 4. Extract와 Insert 함수로 개별 요소 접근
 *
 */

int main() {
    std::cout << "=== SIMD 데이터 접근하기 ===" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // 1. 포인터 변환 (Pointer Conversion)
    // ========================================================================
    std::cout << "1. 포인터 변환 (Pointer Conversion)" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "reinterpret_cast를 사용하여 SIMD 타입과 배열 간 변환합니다." << std::endl;
    std::cout << "간단하지만 잠재적으로 안전하지 않은 방법입니다." << std::endl;
    std::cout << std::endl;

    // 오름차순 값으로 SIMD 벡터 초기화
    __m256 simd_vec1 = _mm256_set_ps(8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f);

    // 포인터 변환을 사용하여 데이터 접근
    // __m256의 주소를 float*로 재해석 → 배열처럼 접근 가능
    float* float_ptr = reinterpret_cast<float*>(&simd_vec1);

    std::cout << "포인터를 통한 SIMD 벡터 값: [";
    for (int i = 0; i < 7; i++) {
        std::cout << float_ptr[i] << ", ";
    }
    std::cout << float_ptr[7] << "]" << std::endl;

    // 포인터를 통해 데이터 수정
    std::cout << "포인터를 통해 값 수정 중..." << std::endl;
    float_ptr[0] = 100.0f;
    float_ptr[4] = 200.0f;

    print_m256(simd_vec1, "수정된 SIMD 벡터");
    std::cout << std::endl;

    // ========================================================================
    // 2. Union 사용
    // ========================================================================
    std::cout << "2. Union 사용" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "Union을 사용하여 SIMD 타입과 배열 간 별칭을 생성합니다." << std::endl;
    std::cout << "포인터 변환보다 깔끔하고 안전한 접근 방식입니다." << std::endl;
    std::cout << std::endl;

    // float SIMD 벡터용 union 정의
    // v와 a는 같은 32바이트 메모리를 공유함
    union FloatSIMD {
        __m256 v;       // SIMD 벡터로 접근
        float a[8];     // 배열로 접근
    };

    // SIMD 벡터로 union 초기화
    FloatSIMD float_union;
    float_union.v = _mm256_set_ps(16.0f, 14.0f, 12.0f, 10.0f, 8.0f, 6.0f, 4.0f, 2.0f);

    // 배열을 통해 데이터 접근
    std::cout << "Union을 통한 SIMD 벡터 값: [";
    for (int i = 0; i < 7; i++) {
        std::cout << float_union.a[i] << ", ";
    }
    std::cout << float_union.a[7] << "]" << std::endl;

    // 배열을 통해 데이터 수정
    std::cout << "Union을 통해 값 수정 중..." << std::endl;
    float_union.a[1] = 42.0f;   // 배열로 수정하면
    float_union.a[6] = 99.0f;   // SIMD 벡터에도 반영됨 (같은 메모리)

    print_m256(float_union.v, "수정된 SIMD 벡터 (union)");

    float8 float8_union;
    float8_union.v = _mm256_set1_ps(5.0f);  // 모든 레인을 5.0으로
    float8_union.a[2] = 10.0f;              // 3번째 index만 10.0으로
    float8_union.a[5] = 20.0f;              // 6번째 index만 20.0으로

    print_m256(float8_union.v, "simd_utils.h의 float8 union 사용");
    std::cout << std::endl;

    // ========================================================================
    // 3. Store와 Load 함수 (권장 방법)
    // ========================================================================
    std::cout << "3. Store와 Load 함수" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "_mm256_store_*와 _mm256_load_* 함수로 데이터를 전송합니다." << std::endl;
    std::cout << "대부분의 상황에서 권장되는 접근 방식입니다." << std::endl;
    std::cout << std::endl;

    // SIMD 벡터 초기화
    __m256 simd_vec3 = _mm256_set_ps(8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f);

    // 정렬된 메모리 할당 (32바이트 정렬)
    // _mm256_store_ps는 정렬된 메모리를 요구함
    float* aligned_array = aligned_alloc<float>(8);

    // SIMD 벡터를 배열에 저장 (레지스터 → 메모리)
    _mm256_store_ps(aligned_array, simd_vec3);

    std::cout << "Store를 통한 SIMD 벡터 값: [";
    for (int i = 0; i < 7; i++) {
        std::cout << aligned_array[i] << ", ";
    }
    std::cout << aligned_array[7] << "]" << std::endl;

    // 배열 수정 (일반 C++ 코드)
    std::cout << "배열 값 수정 중..." << std::endl;
    aligned_array[3] = 30.0f;
    aligned_array[7] = 80.0f;

    // 수정된 배열을 SIMD 벡터로 로드 (메모리 → 레지스터)
    __m256 modified_vec = _mm256_load_ps(aligned_array);

    print_m256(modified_vec, "수정된 SIMD 벡터 (store/load)");
    std::cout << std::endl;

    // ========================================================================
    // 4. Extract와 Insert 요소
    // ========================================================================
    std::cout << "4. Extract와 Insert 요소" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "_mm256_extract_*와 _mm256_insert_* 함수로 개별 요소에 접근합니다." << std::endl;
    std::cout << "몇 개의 요소만 접근할 때 유용합니다." << std::endl;
    std::cout << std::endl;

    // 정수로 SIMD 벡터 초기화
    __m256i simd_int_vec = _mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1);

    // 개별 추출
    // 주의: AVX2에서는 먼저 128비트 레인을 추출한 후, 거기서 추출해야 함
    // __m256i는 2개의 128비트 레인으로 구성됨
    __m128i low_lane = _mm256_extracti128_si256(simd_int_vec, 0);  // 하위 128비트 (요소 0-3)
    __m128i high_lane = _mm256_extracti128_si256(simd_int_vec, 1); // 상위 128비트 (요소 4-7)

    // 128비트 레인에서 32비트 추출
    int element0 = _mm_extract_epi32(low_lane, 0);  // 0 추출
    int element3 = _mm_extract_epi32(low_lane, 3);  // 3 추출
    int element4 = _mm_extract_epi32(high_lane, 0); // 4 추출
    int element7 = _mm_extract_epi32(high_lane, 3); // 7 추출

    std::cout << "추출된 요소들: " << element0 << ", " << element3 << ", "
              << element4 << ", " << element7 << std::endl;

    // 요소 삽입
    // 삽입도 2단계: 128비트 레인 내에서 삽입 → 256비트로 재조립
    __m128i new_low = _mm_insert_epi32(low_lane, 100, 1);  // 1을 100으로 교체
    __m128i new_high = _mm_insert_epi32(high_lane, 200, 2); // 6을 200으로 교체

    // 레인들을 다시 256비트 벡터로 결합
    __m256i modified_int_vec = _mm256_setr_m128i(new_low, new_high);

    print_m256i(modified_int_vec, "수정된 정수 벡터 (extract/insert)");

    free(aligned_array);

    return 0;
}
