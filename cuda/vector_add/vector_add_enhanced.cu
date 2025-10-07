
/*
 * CUDA 에러 처리가 포함된 벡터 덧셈
 * GPU 메모리 할당 실패, 커널 실행 오류 등을 감지
 *
 */

#include <stdio.h>
#include <cuda_runtime.h>

/*
 * cudaCheckError: CUDA API 호출 결과 체크
 *
 * 사용법: cudaCheckError(cudaMalloc(...));
 *
 * 에러 발생 시:
 * - 에러 메시지, 파일명, 라인 번호 출력
 * - 프로그램 종료
 */
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t cuda, const char *file, int line, bool abort=true) {
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/*
 * gpuKernelCheck: 커널 실행 후 에러 체크
 *
 * 사용법:
 *   kernel<<<...>>>();
 *   gpuKernelCheck();
 *
 * 커널 실행 중 발생한 에러 감지:
 * - 잘못된 메모리 접근
 * - 유효하지 않은 커널 설정
 * - 기타 런타임 에러
 */
#define
