/*
 * gpu_info.cu
 *
 * CUDA GPU 정보 조회 프로그램
 * 시스템에 설치된 모든 NVIDIA GPU의 상세 정보를 출력
 *
 */

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    // GPU 개수 조회
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    printf("========================================\n");
    printf("  CUDA 지원 GPU 개수: %d\n", nDevices);
    printf("========================================\n\n");

    // GPU가 없는 경우
    if (nDevices == 0) {
        printf("CUDA 지원 GPU를 찾을 수 없습니다.\n");
        return 1;
    }

    // 각 GPU의 상세 정보 출력
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);  // GPU 속성 조회

        printf("┌─────────────────────────────────────┐\n");
        printf("│  GPU #%d 정보                        │\n", i);
        printf("└─────────────────────────────────────┘\n\n");

        // === 기본 정보 ===
        printf("[ 기본 정보 ]\n");
        printf("  Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        // Compute capability: GPU 아키텍처 버전
        // 7.5 = Turing, 8.0 = Ampere, 8.6 = Ada 등
        printf("\n");

        // === 메모리 정보 ===
        printf("[ 메모리 정보 ]\n");
        printf("  Total global memory: %lu bytes (%.2f GB)\n",
               prop.totalGlobalMem,
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        // 전체 GPU 메모리 크기

        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        // 메모리 클럭 속도 (KHz 단위)

        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        // 메모리 버스 폭 (비트 단위)
        // 일반적으로 256bit, 384bit 등

        // 피크 메모리 대역폭 계산
        // 공식: 2 × Clock(GHz) × BusWidth(bytes) = GB/s
        printf("  Peak Memory Bandwidth (GB/s): %.2f\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        // 2.0 = DDR (Double Data Rate)
        // /8 = bits → bytes 변환
        // /1.0e6 = KHz → GHz 변환
        printf("\n");

        // === 연산 유닛 정보 ===
        printf("[ 연산 유닛 ]\n");
        printf("  Number of SMs: %d\n", prop.multiProcessorCount);
        // SM (Streaming Multiprocessor): GPU의 코어 개수
        // 각 SM은 여러 CUDA 코어를 포함

        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        // 하나의 블록이 가질 수 있는 최대 스레드 수
        // 보통 1024개

        printf("  Max threads dimensions:\n");
        printf("    x = %d, y = %d, z = %d\n",
               prop.maxThreadsDim[0],
               prop.maxThreadsDim[1],
               prop.maxThreadsDim[2]);
        // 블록 내 스레드의 3차원 최대 크기
        // 예: <<<grid, (1024, 1024, 64)>>> 같은 설정의 한계

        printf("  Max grid dimensions:\n");
        printf("    x = %d, y = %d, z = %d\n",
               prop.maxGridSize[0],
               prop.maxGridSize[1],
               prop.maxGridSize[2]);
        // 그리드의 3차원 최대 크기
        // 예: <<<(2147483647, 65535, 65535), threads>>> 같은 설정의 한계
        printf("\n");

        // === 추가 유용한 정보 ===
        printf("[ 추가 정보 ]\n");
        printf("  Warp size: %d\n", prop.warpSize);
        // Warp 크기 (보통 32개 스레드)

        printf("  Shared memory per block: %zu bytes (%.2f KB)\n",
               prop.sharedMemPerBlock,
               prop.sharedMemPerBlock / 1024.0);
        // 블록당 사용 가능한 공유 메모리

        printf("  Registers per block: %d\n", prop.regsPerBlock);
        // 블록당 사용 가능한 레지스터 개수

        printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        // SM당 최대 스레드 수

        printf("  Clock rate: %d KHz (%.2f GHz)\n",
               prop.clockRate,
               prop.clockRate / 1.0e6);
        // GPU 코어 클럭 속도

        printf("\n========================================\n\n");
    }

    printf("[ 요약 ]\n");
    printf("시스템에서 사용 가능한 GPU: %d개\n", nDevices);

    if (nDevices > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("기본 GPU: %s\n", prop.name);
    }

    return 0;
}
