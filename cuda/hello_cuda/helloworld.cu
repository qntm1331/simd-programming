#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void test01()  // GPU에서 실행되는 커널
{
    int warp_ID_Value = threadIdx.x / 32;
    //           스레드 ID ÷ 32 = Warp ID
    //           0~31 → Warp 0
    //           32~63 → Warp 1

    printf("block=%d thread=%d warp=%d\n",
           blockIdx.x,   // 블록 번호 (0 또는 1)
           threadIdx.x,  // 블록 내 스레드 번호 (0~63)
           warp_ID_Value);
}

int main()
{
    test01<<<2, 64>>>();  // 2개 블록, 블록당 64개 스레드
    //     │  └─ 블록당 스레드 수
    //     └──── 블록 개수

    cudaDeviceSynchronize();  // GPU 작업 완료 대기
    return 0;
}
