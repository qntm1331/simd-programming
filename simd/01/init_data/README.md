
## SIMD 벡터를 초기화하는 4가지 방법과 성능 비교

### 1. Zero Initialization (_mm256_setzero_*)

```cpp
__m256 simd_float_vec = _mm256_setzero_ps();
// 결과: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

일반 방식 vs SIMD

```cpp
// 일반 (8번 반복)
for (int i = 0; i < 8; i++) {
    array[i] = 0.0f;
}

// SIMD (1개 명령어로 8개 동시에)
vec = _mm256_setzero_ps();
```

### 2. Broadcast Initialization (_mm256_set1_*)

```cpp
__m256 vec = _mm256_set1_ps(42.0f);
// 결과: [42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0]
```

- 모든 레인에 같은 값을 넣을 때 사용. 반복문 없이 한 번에 8개를 같은 값으로 설정.

### 3. Individual Initialization (_mm256_set_*)

```cpp
__m256i vec = _mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1);
// 주의: 역순! 결과: [1, 2, 3, 4, 5, 6, 7, 8]
```

- 첫 번째 인자(8)가 마지막 레인(인덱스 7)에 들어감
- 마지막 인자(1)가 첫 번째 레인(인덱스 0)에 들어감

### 4. Reverse Order Initialization (_mm256_setr_*)

```cpp
__m256 vec = _mm256_setr_ps(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
// 결과: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
```

- 첫 번째 인자가 첫 번째 레인에
- 마지막 인자가 마지막 레인에

---

### 성능 비교


```cpp
// 100만 번 반복해서 시간 측정
for (int i = 0; i < 1000000; i++) {
    // 일반 방식
    for (int j = 0; j < 8; j++) {
        array[j] = value;
    }
}
// vs
for (int i = 0; i < 1000000; i++) {
    // SIMD 방식 (1개 명령어)
    vec = _mm256_set1_ps(value);
}
```

---

### 사용 예시

```cpp
// 배열 0으로 초기화
__m256 zeros = _mm256_setzero_ps();
for (int i = 0; i < array_size; i += 8) {
    _mm256_storeu_ps(&array[i], zeros);
}

// 배열을 특정 값으로 채우기
__m256 fill_value = _mm256_set1_ps(3.14f);
for (int i = 0; i < array_size; i += 8) {
    _mm256_storeu_ps(&array[i], fill_value);
}

// 1, 2, 3, 4, 5, 6, 7, 8로 초기화
__m256 sequence = _mm256_setr_ps(1, 2, 3, 4, 5, 6, 7, 8);
```

---

### Summery

타입별 함수 구분

- _ps: float
- _pd: double
- _epi32: int32
- _epi16: int16

성능 차이

- Zero/Broadcast: 매우 빠름 (1개 명령어)
- Individual: 약간 느림 (여러 명령어 조합)
