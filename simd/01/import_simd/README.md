## SIMD (Single Instruction Multiple Data)

일반연산: 1 + 1 = 2
SIMD 연산: [1, 2, 3, 4] + [5, 6, 7, 8] = [6,8,10,12]

### 1. Helper function

- print_m256

```cpp
void print_m256(const __m256& v, const char * label) {
    float result[8];
    _mm256_storeu_ps(result, v); // SIMD register -> 일반 배열로 복사
```

- __m256: 256비트 레지스터 (32비트 float X 8개)
- _mm256_storeu_ps: SIMD 값을 일반 배열로 복사

### 2. Example 1: Float 연산 (8개 동시 처리)

```cpp
__m256 a = _mm256_set_ps(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
__m256 b = _mm256_set_ps(8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f);
```

메모리 구조 (역순으로 저장)

```cpp
a: [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
b: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

__m256 sum = _mm256_add_ps(a, b);  // 8개를 한 번에 더함!
```

일반 방식 vs SIMD:

```cpp
// 일반 방식 (8번 반복)
for (int i = 0; i < 8; i++) {
    result[i] = a[i] + b[i];
}

// SIMD 방식 (1번 실행으로 8개 처리!)
__m256 sum = _mm256_add_ps(a, b);
```

### 3. Example 2: Integer 연산 (8개 동시)

```cpp
__m256i int_a = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
//                               ↑ 역순으로 저장: [8,7,6,5,4,3,2,1]
```
- __m256i: 정수용 256비트 레지스터 (32비트 int × 8개)
- _mm256_add_epi32: 8개 정수를 동시에 더함
- _mm256_mullo_epi32: 8개 정수를 동시에 곱함

### 4. Example 3: Double 연산 (4개 동시)

```cpp
__m256d double_a = _mm256_set_pd(1.0, 2.0, 3.0, 4.0);
```
- __m256d: Double용 256비트 레지스터 (64비트 double × 4개)
- Double은 크기가 2배라서 4개만 들어감

### 5. Example 4: 배열 합계

```cpp
float numbers[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

__m256 vec1 = _mm256_loadu_ps(&numbers[0]);   // 앞 8개 로드
__m256 vec2 = _mm256_loadu_ps(&numbers[8]);   // 뒤 8개 로드
__m256 total = _mm256_add_ps(vec1, vec2);     // 8쌍 동시에 더하기
```

```
vec1:  [1,  2,  3,  4,  5,  6,  7,  8]
vec2:  [9, 10, 11, 12, 13, 14, 15, 16]
       --------------------------------
total: [10, 12, 14, 16, 18, 20, 22, 24]
```

Horizontal Sum (가로 합계)

```cpp
// total = [10, 12, 14, 16, 18, 20, 22, 24]
// → 이걸 하나의 값으로 합치기

__m128 high = _mm256_extractf128_ps(total, 1);  // 상위 4개
__m128 low = _mm256_castps256_ps128(total);     // 하위 4개
__m128 sum128 = _mm_add_ps(high, low);          // [28, 32, 36, 40]
sum128 = _mm_hadd_ps(sum128, sum128);           // [60, 76, ...]
sum128 = _mm_hadd_ps(sum128, sum128);           // [136, ...]
```

--

### 성능 비교

```cpp
// 일반 코드 (16번 더하기)
for (int i = 0; i < 16; i++) sum += numbers[i];  // 16회 반복

// SIMD 코드 (2번 + 알파)
vec1 + vec2  // 8쌍 동시 처리
+ horizontal sum  // 추가 연산
```
- 이론상: 최대 8배 빠름
- 실제: 데이터 크기와 패턴에 따라 2~6배 정도

---

### 주요 함수

- _mm256_set_ps : 8개 float 값으로 레지스터 초기화
- _mm256_add_ps : 8개 float 동시 덧셈
- _mm256_mul_ps : 8개 float 동시 곱셈
- _mm256_loadu_ps : 메모리에서 8개 float 로드 (정렬 불필요)
- _mm256_storeu_ps : 레지스터를 메모리에 저장
- _mm256_hadd_ps : 인접한 값들을 더함 (horizontal add)

접미사 의미

- _ps: Packed Single (float)
- _pd: Packed Double
- _epi32: 32bit integer

