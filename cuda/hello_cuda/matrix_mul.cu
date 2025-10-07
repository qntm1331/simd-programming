#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// 타일 크기 (32 * 32)
// Warp 크기(32)의 배수로 설정하여 효율 극대화
#define TILE_WIDTH 32

/*
 * CUDA 커널: Tiled 행렬 곱셈
 *
 * 최적화 전략:
 * 1. Global Memory → Shared Memory로 타일 단위 복사
 * 2. Shared Memory에서 반복 계산 (빠른 접근)
 * 3. 타일 단위로 순회하며 부분 결과 누적
 *
 * 파라미터:
 *   a, b: 입력 행렬 (GPU 메모리)
 *   c: 출력 행렬 (GPU 메모리)
 *   width: 행렬 크기 (정사각 행렬)
 */
__global__ void matrixMul(int *a, int *b, int *c, int width) {
    // Shared Memory 선언
    // 블록 내 모든 스레드가 공유하는 고속 메모리
    // 32 X 32 타일 2개 (A용, B용)
    __shared__ int ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ int ds_B[TILE_WIDTH][TILE_WIDTH];

    // thread index 계산
    int bx = blockIdx.x; // block X 좌표
    int by = blockIdx.y; // block Y 좌표
    int tx = threadIdx.x; // block 내 thread X 좌표
    int ty = threadIdx.y; // block 내 thread Y 좌표

    // 전역 행렬에서 이 thread 가 계산할 위치
    int Row = by * TILE_WIDTH + ty; // 결과 행렬 C의 행 번호
    int Col = bx * TILE_WIDTH + tx; // 결과 행렬 C의 열 번호

    // 이 thread가 계산할 C[Row][Col]의 누적값
    int Cvalue = 0;

    // 타일 단위 순회
    // 행렬을 TILE_WIDTH * TILE_WIDTH 크기의 타일로 나눔
    // 전체 타일 개수 = width / TILE_WIDTH
    for (int t = 0; t < width / TILE_WIDTH; ++t) {
        // 1. Global → Shared Memory 복사
        // A 행렬의 t번째 타일을 Shared Memory로 복사
        // A[Row][t*TILE_WIDTH + tx]
        ds_A[ty][tx] = a[Row * width + t * TILE_WIDTH + tx];

        // B 행렬의 t번째 타일을 Shared Memory로 복사
        // B[t*TILE_WIDTH + ty][Col]
        ds_B[ty][tx] = b[(t * TILE_WIDTH +ty) * width + Col];

        // 동기화
        // 블록 내 모든 스레드가 복사 완료할 때까지 대기
        // 모든 thread가 ds_A, ds_B 를 안전하게 읽을 수 있도록 보장
        __syncthreads();

        // 2. Shared Memory 에서 계산
        // 타일 내에서 내적 계산
        // Cvalue += A의 행 * B의 열
        for (int k = 0;k < TILE_WIDTH; ++k) {
            Cvalue += ds_A[ty][k] * ds_B[k][tx];
            // ds_A[ty][k]: A 행렬의 현재 행에서 k번째 요소
            // ds_B[k][tx]: B 행렬의 k번째 행, 현재 열
        }

        // 동기화
        // 다음 타일로 넘어가기전에 모든 계산 완료대기
        // ds_A, ds_B를 덮어쓰기 전에 모든 thread 가 사용 완료하도록 보장
        __syncthreads();

    }

    // 결과를 global memory에 저장
    // 모든 타일 순회 후 최종값 저장
    c[Row * width + Col] = Cvalue;
}

void matrixMulCPU(int *a, int *b, int *c, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            int sum = 0;
            for (int k = 0; k < width; k++) {
                sum += a[i * width + k] * b[k * width + j];
            }
            c[i * width + j] = sum;
        }
    }
}

/*
 * 행렬 초기화
 *
 * A[i][j] = i * width + j
 * B[i][j] = j * width + i (A의 전치 패턴)
 * C[i][j] = 0 (초기화)
 */
void matrixInit(int *a, int *b, int *c, int width) {
    for (int i = 0; i < width; ++ i) {
        for (int j = 0; j < width; ++j) {
            a[i * width + j] = i * width + j;
            b[i * width + j] = j * width + i;
            c[i * width + j] = 0;
        }
    }
}

long long getCurrentTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000000LL + (long long)tv.tv_usec;
}

int main() {
    int *a, *b, *c; // 호스트(CPU) 메모리
    int *d_a, *d_b, *d_c; // 디바이스(GPU) 메모리

    // 행렬 크기: 16384 × 16384 (4096 * 4)
    int width = 4096 * 4;
    int size = width * width * sizeof(int);

    printf("========================================\n");
    printf("  Tiled 행렬 곱셈 (Shared Memory 사용)\n");
    printf("========================================\n");
    printf("행렬 크기: %d × %d\n", width, width);
    printf("메모리 사용량: %.2f GB (행렬당)\n", size / (1024.0 * 1024.0 * 1024.0));
    printf("타일 크기: %d × %d\n", TILE_WIDTH, TILE_WIDTH);
    printf("\n");

    // host memory 할당
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    if (!a || !b || !c) {
        printf("CPU 메모리 할당 실패\n");
        return 1;
    }

    // GPU Memory 할당
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // 행렬 초기화
    matrixInit(a, b, c, width);

    // 데이터 복사 CPU -> GPU
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // kernel 설정
    // block: 32 * 32 thread (1024 개 thread)
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

    // grid: 필요한 block 개수 계산(올림 나눗셈)
    // 16384 ÷ 32 = 512개 블록 (가로, 세로 각각)
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH,
            (width + TILE_WIDTH - 1) / TILE_WIDTH);

    printf("\n커널 설정:\n");
    printf("  블록: %d × %d 스레드\n", TILE_WIDTH, TILE_WIDTH);
    printf("  그리드: %d × %d 블록\n", dimGrid.x, dimGrid.y);
    printf("  총 스레드: %d개\n", dimGrid.x * dimGrid.y * TILE_WIDTH * TILE_WIDTH);

    // kernel 실행 (시간 측정)
    printf("\n행렬 곱셈 실행 중...\n");

    long long start_time = getCurrentTime();

    // kernel 실행
    matrixMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, width);

    // GPU 작업 완료 대기
    cudaDeviceSynchronize();

    long long end_time = getCurrentTime();

    // 실행 시간 계산
    double execution_time = (double)(end_time - start_time) / 1000000.0;

    printf("실행 시간: %.6f 초\n", execution_time);

    // FLOPS 계산 (Floating Point Operations Per Second)
    // 행렬 곱셈: width^3 * 2 (곱셈 + 덧셈)
    long long operations = 2LL * width * width * width;
    double gflops = (operations / execution_time) / 1e9;
    printf("성능: %.2f GFLOPS\n", gflops);

    // 결과 복사: GPU → CPU
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // 결과 검증 (일부)
    printf("\n결과 검증 (C[0][0] ~ C[2][2]):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%10d ", c[i * width + j]);
        }
        printf("\n");
    }

    // === 메모리 해제 ===
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    return 0;
}

