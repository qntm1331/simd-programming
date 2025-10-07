
## Memory 구조

- CPU와 GPU는 완전히 분리된 메모리 공간을 사용한다.
    ```
    CPU 메모리          GPU 메모리
    ┌─────────┐        ┌─────────┐
    │  A[0]   │        │         │
    │  A[1]   │  복사→ │  d_A[0] │
    │  A[2]   │  ====> │  d_A[1] │
    │   ...   │        │  d_A[2] │
    └─────────┘        │   ...   │
                       └─────────┘
    ```

```
CPU (호스트)
  RAM ← 여기에 int *A, *B, *C

     ↕ cudaMemcpy로 복사

GPU (디바이스)
  ┌─────────────────────────┐
  │ Global Memory (전역)    │ ← cudaMalloc으로 할당 (d_A, d_B, d_C)
  │ - 크기: 수 GB           │    모든 스레드가 접근 가능
  │ - 속도: 느림            │    하지만 가장 느림
  ├─────────────────────────┤
  │ Shared Memory (공유)    │ ← __shared__ 선언
  │ - 크기: 블록당 48KB     │    블록 내 스레드만 공유
  │ - 속도: 100배 빠름      │
  ├─────────────────────────┤
  │ Registers (레지스터)    │ ← int i = ... 같은 지역변수
  │ - 크기: 매우 작음       │    각 스레드 전용
  │ - 속도: 가장 빠름       │
  └─────────────────────────┘
```


### cudaMemcpy(dst, src, size, direction);

- cudaMemcpyHostToDevice   // CPU → GPU (입력 데이터)
- cudaMemcpyDeviceToHost   // GPU → CPU (결과 받기)
- cudaMemcpyDeviceToDevice // GPU → GPU (같은 GPU 내)
- cudaMemcpyHostToHost     // CPU → CPU (거의 안 씀)

### GPU Memory Hierachy

```
GPU 칩 내부:
┌────────────────────────────────────────┐
│                                        │
│  ┌──────────────────────────────┐      │
│  │    Global Memory (VRAM)      │      │ ← GPU 보드의 GDDR6 메모리
│  │    크기: 8GB ~ 80GB          │      │   (GPU 외부이지만 전용)
│  └──────────────────────────────┘      │
│                                        │
│  SM 0          SM 1          SM 2      │
│  ┌────┐       ┌────┐       ┌────┐      │
│  │Reg │       │Reg │       │Reg │      │ ← 레지스터
│  │Reg │       │Reg │       │Reg │      │   (SM 내부, 빠름)
│  ├────┤       ├────┤       ├────┤      │
│  │Shar│       │Shar│       │Shar│      │ ← Shared Memory
│  │ed  │       │ed  │       │ed  │      │   (SM 내부, 빠름)
│  └────┘       └────┘       └────┘      │
│                                        │
│  ... (수십~수백 개 SM)                 │
└────────────────────────────────────────┘
```

Global Memory (전역 메모리)
- 위치: GPU 보드의 GDDR6/HBM 메모리 칩
- 크기:
    - RTX 3090: 24GB
    - RTX 4090: 24GB
    - A100: 40GB/80GB
- 특징
    ```cu
    cudaMalloc(&d_A, size);  // 여기 할당됨

    __global__ void kernel(int *d_A) {
        d_A[i] = 10;  // 모든 스레드가 접근 가능
    }
    ```
    - 모든 블록, 모든 스레드가 접근 가능
    - 가장 느림 (400~900 GB/s)
    - 가장 큼 (수십 GB)
    - 물리적 위치: GPU 카드의 메모리 칩 (CPU RAM과는 별개)

Shared Memory (공유 메모리)
- 위치: 각 SM(Streaming Multiprocessor) 내부
- 크기: 블록당 48KB~164KB
- 특징
    ```cu
    __global__ void kernel() {
        __shared__ int ds_A[32][32];  // 여기 할당됨

        // 같은 블록 내 스레드만 공유
        ds_A[ty][tx] = 10;
        __syncthreads();  // 동기화 필요
    }
    ```
    - 블록 내 스레드들만 공유
    - Global Memory보다 100배 빠름 (수십 TB/s)
    - 크기 제한 있음 (블록당 수십 KB)
    - 물리적 위치: SM(코어) 내부의 SRAM

Registers (레지스터)
- 위치: 각 SM 내부의 레지스터 파일
- 크기: 스레드당 255개 (SM당 65,536개)
- 특징
    ```cu
    __global__ void kernel() {
        int i = threadIdx.x;  // 레지스터에 저장
        int sum = 0;          // 레지스터에 저장
        float x = 1.5f;       // 레지스터에 저장
    }
    ```
    - 각 스레드 전용 (다른 스레드 접근 불가)
    - 가장 빠름 (1 사이클)
    - 가장 작음 (스레드당 ~1KB)
    - 물리적 위치: SM 내부의 레지스터 파일

실제 하드웨어 구조
```
RTX 3090 예시:
┌────────────────────────────────────┐
│ GPU 칩                             │
│                                    │
│ 82개 SM × 각 SM마다:               │
│   - 레지스터: 65,536개             │
│   - Shared Memory: 100KB           │
│                                    │
└────────────────────────────────────┘
         ↕ PCIe 버스
┌────────────────────────────────────┐
│ GDDR6X 메모리 (24GB)               │ ← Global Memory
│ 대역폭: 936 GB/s                   │
└────────────────────────────────────┘
```


