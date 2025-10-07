/*
 * CUDA 벡터 덧셈 프로그램
 * 두 개의 큰 벡터(각 32M 요소)를 GPU에서 병렬로 더함
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define SIZE 1024*1024*32

/*
 * CUDA 커널: 벡터 덧셈
 *
 * 각 스레드가 벡터의 한 요소를 담당
 * C[i] = A[i] + B[i]
 *
 * 파라미터:
 *   A, B: 입력 벡터 (GPU 메모리)
 *   C: 출력 벡터 (GPU 메모리)
 *   n: 벡터 크기
 */
__global__ void vectorAdd(int *A, int *B, int *C, int n) {
    // 전역 thread ID 계산
    // 이 thread 가 처리할 배열 인덱스
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    // 배열 범위를 벗어나지 않도록 체크
    // 마지막 블록은 일부 스레드만 사용
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;
    int size = SIZE * sizeof(int);

    // CUDA 이벤트 생성
    // GPU timer (cudaEventElapsedTime으로 정확한 커널 실행 시간 측정)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // host memory 할당
    A = (int *)malloc(size);
    B = (int *)malloc(size);
    C = (int *)malloc(size);

    // vector 초기화
    // A[i] = i (0, 1, 2, 3, ...)
    // B[i] = SIZE - i (33554432, 33554431, 33554430, ...)
    for (int i = 0; i < SIZE; i++) {
        A[i] = i;
        B[i ] = SIZE - i;
    }
    // 예상 결과: C[i] = i + (SIZE - i) = SIZE (모든 요소가 33554432)

    // GPU memory 할당
    // cudaMalloc: GPU 전역 메모리에 공간 할당
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // data 복사 cpu -> gpu
    // cudaMemcpyHostToDevice: host -> device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // timer start
    cudaEventRecord(start);

    // kernel 실행 설정
    int threadsPerBlock = 96; // 블록당 96개 스레드

    // 블록 개수 계산 (올림 나눗셈)
    // 모든 요소를 처리하기 위해 필요한 블록 수
    // (33554432 + 96 - 1) / 96 = 349526 블록
    int blocksPerGrid = (SIZE + threadsPerBlock - 1) / threadsPerBlock;

    // 커널 실행
    // <<<블록 개수, 블록당 스레드 개수>>>
    // 총 스레드: 349526 * 96 = 33,554,496 (SIZE보다 약간 많음)
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, SIZE);

    // 타이머 종료
    // 커널 실행 완료
    cudaEventRecord(stop);

    // 결과 복사: GPU → CPU
    // 계산 결과를 CPU 메모리로 가져옴
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // 실행 시간 계산
    // GPU 작업이 완전히 끝날 때까지 대기
    cudaEventSynchronize(stop);

    // start와 stop 사이의 시간 계산 (밀리초)
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("=== CUDA 벡터 덧셈 ===\n");
    printf("벡터 크기: %d 개 요소\n", SIZE);
    printf("메모리 사용: %.2f MB (벡터당)\n", size / (1024.0 * 1024.0));
    printf("블록: %d개\n", blocksPerGrid);
    printf("블록당 스레드: %d개\n", threadsPerBlock);
    printf("총 스레드: %d개\n", blocksPerGrid * threadsPerBlock);
    printf("\n실행 시간: %f 밀리초\n", milliseconds);

    // 결과 검증 (처음 10개 출력)
    printf("\n처음 10개 요소:\n");
    for(int i = 0; i < 10; i++) {
        printf("A=%-8d\tB=%-8d\t--->\tC=%-8d", A[i], B[i], C[i]);

        // 검증: C[i] = SIZE 여야 함
        if (C[i] == SIZE) {
            printf(" ✓\n");
        } else {
            printf(" ✗ (예상: %d)\n", SIZE);
        }
    }

    // GPU 메모리 해제
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // CPU 메모리 해제
    free(A);
    free(B);
    free(C);

    // CUDA 이벤트 삭제
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
