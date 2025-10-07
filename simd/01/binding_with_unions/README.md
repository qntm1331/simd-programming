
##  SIMD 벡터의 데이터에 접근하는 4가지 방법

SIMD 레지스터(__m256)는 특수한 타입이라 일반 배열처럼 vec[0], vec[1] 같은 방식으로 접근할 수 없다.
데이터를 읽거나 수정하려면 특별한 방법이 필요하다.

### 1. 포인터 변환 (reinterpret_cast)

```cpp
__m256 simd_vec = _mm256_set_ps(8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f);
float* float_ptr = reinterpret_cast<float*>(&simd_vec);

float_ptr[0] = 100.0f;  // 첫 번째 요소 수정
```

- __m256 변수의 주소를 float*로 재해석
- 메모리는 같은데 타입만 바꿔서 배열처럼 접근
- 장점: 간단하고 직관적
- 단점: 타입 안전성 없음, 정렬 문제 가능

### 2. Union 사용

한 번의 메모리 할당으로 SIMD 타입과 배열 타입 모두로 사용 가능.
메모리 복사 없이 단지 "해석 방법"만 바꾸는 것
```cpp
union FloatSIMD {
    __m256 v;      // SIMD 벡터
    float a[8];    // 일반 배열
};

FloatSIMD data;
data.v = _mm256_set_ps(...);  // SIMD로 초기화
data.a[0] = 100.0f;           // 배열로 접근


// SIMD 연산은 v로
data.v = _mm256_add_ps(x, y);

// 개별 접근은 a로
if (data.a[3] > 100.0f) {
    data.a[3] = 100.0f;
}

// 다시 SIMD 연산
result = _mm256_mul_ps(data.v, scale);
```

- Union은 같은 메모리를 여러 타입으로 공유
- v와 a가 같은 32바이트 메모리를 가리킴

메모리 레이아웃

```cpp
+---+---+---+---+---+---+---+---+
| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |  ← a[8]
+---+---+---+---+---+---+---+---+
|          __m256 v              |
```

- 장점: 더 안전하고 명시적
- 단점: Union 자체의 타입 안전성 제한

### 3. Store/Load 함수 (권장)

```cpp
__m256 vec = _mm256_set_ps(8, 7, 6, 5, 4, 3, 2, 1);

// SIMD → 배열
float* array = aligned_alloc<float>(8);
_mm256_store_ps(array, vec);

// 배열 수정
array[3] = 30.0f;

// 배열 → SIMD
vec = _mm256_load_ps(array);
```

작동 원리

- _mm256_store_ps: SIMD 레지스터 → 메모리
- _mm256_load_ps: 메모리 → SIMD 레지스터
- 명시적인 데이터 전송

aligned_alloc이 필요한 이유

- _mm256_store_ps는 32바이트 정렬 필요
- 정렬되지 않으면 크래시 발생
- _mm256_storeu_ps는 정렬 불필요

- 장점: 명확하고 안전, 최적화 가능
- 단점: 코드가 약간 길어짐

### 4. Extract/Insert (개별 요소)

```cpp
__m256i vec = _mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1);

// 추출: 256비트 → 128비트 → 32비트
__m128i low = _mm256_extracti128_si256(vec, 0);   // 하위 4개
int val = _mm_extract_epi32(low, 0);              // 첫 번째 요소

// 삽입: 32비트 → 128비트 → 256비트
__m128i new_low = _mm_insert_epi32(low, 100, 1);  // 요소 1 변경
vec = _mm256_setr_m128i(new_low, high);           // 재조립
```

- AVX2는 256비트에서 직접 32비트 요소 접근 불가
- 128비트 레인으로 나눠서 접근해야 함

레인 구조

```bash
__m256i (256비트)
├─ low  (128비트): [0][1][2][3]
└─ high (128비트): [4][5][6][7]
```

- 장점: 1-2개 요소만 수정할 때 효율적
- 단점: 여러 요소 접근 시 복잡하고 느림

---

### 사용 시나리오

전체 벡터 처리 → Store/Load
```cpp
__m256 result = _mm256_add_ps(a, b);
float output[8];
_mm256_storeu_ps(output, result);
```

디버깅/출력 → Union

```cpp
float8 debug;
debug.v = some_simd_vector;
for (int i = 0; i < 8; i++) {
    printf("%f ", debug.a[i]);
}
```

특정 요소만 수정 → Extract/Insert

```cpp
// 벡터의 첫 번째 요소만 0으로 변경
__m128i low = _mm256_extracti128_si256(vec, 0);
low = _mm_insert_epi32(low, 0, 0);
vec = _mm256_inserti128_si256(vec, low, 0);
```

빠른 프로토타입 → reinterpret_cast

```cpp
float* ptr = reinterpret_cast<float*>(&vec);
// 주의: 프로덕션 코드에는 권장 안 함
```

### Store/Load 과정

CPU는 데이터를 효율적으로 읽기 위해 특정 주소 경계에서 읽음

Store/Load는 레지스터와 메모리 간의 명시적인 데이터 복사.

Union과 달리 데이터가 실제로 이동하지만, 정렬된 메모리를 사용하면 매우 빠르게 처리.

```cpp
// 정렬되지 않은 메모리
float array[8];  // 주소가 0x1003 같은 임의의 위치일 수 있음

// 32바이트 정렬된 메모리
float* aligned = aligned_alloc<float>(8);  // 주소가 0x1000, 0x1020 등
```

정렬된 주소
```
좋은 주소 (32바이트 정렬):
0x0000, 0x0020, 0x0040, 0x0060 ...
↑ 32(0x20)의 배수

나쁜 주소 (정렬 안 됨):
0x0003, 0x0017, 0x002B ...
```

#### Store/Load 과정

- 1. SIMD 레지스터 → 메모리 (Store)
    ```
    __m256 vec = _mm256_set_ps(8, 7, 6, 5, 4, 3, 2, 1);
    float* array = aligned_alloc<float>(8);
    _mm256_store_ps(array, vec);
    ```
    메모리 변화
    ```
        [레지스터]                    [메모리]
    __m256 vec                    float array[8]
    ┌────────────┐                ┌─────┐
    │ 1 2 3 4    │  store_ps      │ 1.0 │ array[0] (주소: 0x1000)
    │ 5 6 7 8    │  ========>     │ 2.0 │ array[1] (주소: 0x1004)
    └────────────┘                │ 3.0 │ array[2] (주소: 0x1008)
      (256비트)                   │ 4.0 │ array[3] (주소: 0x100C)
                                  │ 5.0 │ array[4] (주소: 0x1010)
                                  │ 6.0 │ array[5] (주소: 0x1014)
                                  │ 7.0 │ array[6] (주소: 0x1018)
                                  │ 8.0 │ array[7] (주소: 0x101C)
                                  └─────┘
                                   (32바이트)
    ```
    CPU가 하는 일
    - 레지스터의 256비트를 한 번에 메모리에 쓰기
    - 32바이트 정렬된 주소라서 1번의 메모리 쓰기로 완료
- 2. 메모리 수정
    ```
    array[3] = 30.0f;
    array[7] = 80.0f;
    ```
    메모리 상태
    ```
    주소      값
    0x1000:  1.0   array[0]
    0x1004:  2.0   array[1]
    0x1008:  3.0   array[2]
    0x100C:  30.0  array[3] ← 수정
    0x1010:  5.0   array[4]
    0x1014:  6.0   array[5]
    0x1018:  7.0   array[6]
    0x101C:  80.0  array[7] ← 수정
    ```
- 3. 메모리 → SIMD 레지스터 (Load)
    ```cpp
    __m256 modified = _mm256_load_ps(array);
    ```
    메모리에서 레지스터로
    ```
    [메모리]                      [레지스터]
    float array[8]                __m256 modified
    ┌─────┐                      ┌────────────┐
    │ 1.0 │ 0x1000               │ 1 2 3 30   │
    │ 2.0 │ 0x1004    load_ps    │ 5 6 7 80   │
    │ 3.0 │ 0x1008   ========>   └────────────┘
    │ 30.0│ 0x100C                 (256비트)
    │ 5.0 │ 0x1010
    │ 6.0 │ 0x1014
    │ 7.0 │ 0x1018
    │ 80.0│ 0x101C
    └─────┘
    ```
    CPU가 하는 일
    - 메모리의 32바이트를 한 번에 레지스터로 읽기
    - 정렬되어 있어서 1번의 메모리 읽기로 완료


#### 정렬된 vs 정렬 안 된 접근

정렬된 접근 (_mm256_store_ps)

```
메모리 주소: 0x1000 (32바이트 정렬)
┌──────────────────────────────────┐
│  32바이트를 한 번에 읽기/쓰기     │
└──────────────────────────────────┘
CPU 버스 폭과 딱 맞음 → 빠름
```

정렬 안 된 접근 (_mm256_storeu_ps)

```
메모리 주소: 0x1003 (정렬 안 됨)
   ┌──────────────────────────────────┐
   │  경계를 넘어감                   │
   └──────────────────────────────────┘
┌─────┐                           ┌─────┐
│ 1번 │                           │ 2번 │
└─────┘                           └─────┘
2번의 메모리 접근 필요 → 느림
```


#### 실제 어셈블리 레벨

```cpp
_mm256_store_ps(array, vec);
```

생성되는 어셈블리

```
vmovaps ymm0, [rdi]    ; 1개 명령어 (aligned)
```

```cpp
_mm256_storeu_ps(array, vec);
```

생성되는 어셈블리

```
vmovups ymm0, [rdi]    ; 1개 명령어지만 내부적으로 느림 (unaligned)
```

#### 전체 데이터 흐름

```
1. SIMD 연산
   __m256 result = _mm256_add_ps(a, b);
   [레지스터] 1 2 3 4 5 6 7 8

2. Store (레지스터 → 메모리)
   _mm256_store_ps(array, result);
   [메모리] 0x1000: 1 2 3 4 5 6 7 8

3. 일반 코드로 수정
   array[3] = 100;
   [메모리] 0x1000: 1 2 3 100 5 6 7 8

4. Load (메모리 → 레지스터)
   __m256 modified = _mm256_load_ps(array);
   [레지스터] 1 2 3 100 5 6 7 8

5. 다시 SIMD 연산
   result = _mm256_mul_ps(modified, scale);
```
