/*
 * vector_add.cu
 *
 * CUDA를 사용한 벡터 덧셈 예제
 * 두 개의 큰 벡터를 GPU에서 병렬로 더하는 프로그램
 *
 */

#include <stdio.h>
#include <cuda_runtime.h>

// 벡터 크기 정의: 32MB (1024 * 1024 * 32 개의 int)
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
 *   n: 벡터의 크기
 */
__global__ void vectorAdd(int *A, int *B, int *C, int n) {
    // 전역 스레드 ID 계산
    // threadIdx.x: 블록 내 스레드 번호
    // blockDim.x: 블록당 스레드 개수
    // blockIdx.x: 블록 번호
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    // 배열 범위를 벗어나지 않도록 체크
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // === 1. 변수 선언 ===
    int *A, *B, *C;            // 호스트(CPU) 메모리 포인터
    int *d_A, *d_B, *d_C;      // 디바이스(GPU) 메모리 포인터
    int size = SIZE * sizeof(int);  // 바이트 단위 크기

    // CUDA 이벤트 생성 (실행 시간 측정용)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // === 2. 호스트 메모리 할당 및 초기화 ===
    printf("벡터 크기: %d 개 (%.2f MB)\n", SIZE, size / (1024.0 * 1024.0));

    A = (int *)malloc(size);
    B = (int *)malloc(size);
    C = (int *)malloc(size);

    // 벡터 초기화
    // A[i] = i, B[i] = SIZE - i
    for (int i = 0; i < SIZE; i++) {
        A[i] = i;
        B[i] = SIZE - i;
    }


    // === 3. GPU 메모리 할당 ===
    printf("GPU 메모리 할당 중...\n");
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);


    // === 4. 데이터를 CPU에서 GPU로 복사 ===
    printf("데이터 복사: CPU -> GPU\n");
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);


    // === 5. 커널 실행 설정 ===
    int threadsPerBlock = 96;  // 블록당 96개 스레드
    // 블록 개수 계산: 올림 나눗셈 (모든 요소를 커버하도록)
    int blocksPerGrid = (SIZE + threadsPerBlock - 1) / threadsPerBlock;

    printf("\n커널 설정:\n");
    printf("  블록당 스레드: %d\n", threadsPerBlock);
    printf("  총 블록 개수: %d\n", blocksPerGrid);
    printf("  총 스레드: %d\n", blocksPerGrid * threadsPerBlock);


    // === 6. 커널 실행 (시간 측정) ===
    cudaEventRecord(start);  // 타이머 시작

    // 커널 실행
    // <<<블록 개수, 블록당 스레드 개수>>>
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, SIZE);

    cudaEventRecord(stop);   // 타이머 종료


    // === 7. 결과를 GPU에서 CPU로 복사 ===
    printf("\n데이터 복사: GPU -> CPU\n");
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);


    // === 8. 실행 시간 계산 및 출력 ===
    cudaEventSynchronize(stop);  // GPU 작업 완료 대기

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("\n=== 결과 ===\n");
    printf("실행 시간: %.3f ms\n", milliseconds);
    printf("처리량: %.2f GB/s\n",
           (3.0 * size) / (milliseconds / 1000.0) / (1024.0 * 1024.0 * 1024.0));
    // 3.0 = A 읽기 + B 읽기 + C 쓰기


    // === 9. 결과 검증 (처음 10개 출력) ===
    printf("\n처음 10개 요소:\n");
    printf("%-10s %-10s %-10s %-10s\n", "Index", "A", "B", "C (A+B)");
    printf("--------------------------------------------\n");
    for(int i = 0; i < 10; i++) {
        printf("%-10d %-10d %-10d %-10d", i, A[i], B[i], C[i]);

        // 검증: C[i]가 A[i] + B[i]와 같은지 확인
        if (C[i] == A[i] + B[i]) {
            printf(" ✓\n");
        } else {
            printf(" ✗ 오류!\n");
        }
    }


    // === 10. 메모리 해제 ===

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

    printf("완료!\n");

    return 0;
}
