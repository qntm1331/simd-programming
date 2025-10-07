#include <bits/stdc++.h>
#include <omp.h>

// 계산할 총 스텝 수
static long num_steps = 100000000;

// False Sharing 방지를 위한 패딩 크기
// 일반적인 캐시 라인 크기가 64바이트 = 8개의 long(8바이트)
#define PAD 8

int main() {
    // 변수 선언
    int i, x, total, nthreads, constant;
    constant = 12;
    total = 0;

    // === 순차 실행 (비교용) ===
    printf("=== 순차 실행 ===\n");
    double start_seq = omp_get_wtime();  // 시작 시간 측정

    int total_seq = 0;
    for (i = 0; i < num_steps; i++) {
        x = i + constant;
        total_seq += x;
    }
    total_seq *= constant;  // 최종 계산

    double end_seq = omp_get_wtime();  // 종료 시간 측정
    double time_seq = end_seq - start_seq;

    printf("결과: %d\n", total_seq);
    printf("실행 시간: %.6f 초\n\n", time_seq);


    // === 병렬 실행 ===
    printf("=== 병렬 실행 (OpenMP) ===\n");

    // 사용할 스레드 개수 설정 (4개)
    omp_set_num_threads(4);

    // 병렬 실행 시작 시간 측정
    double start_par = omp_get_wtime();

    // 병렬 영역 시작
    // 이 블록 내부 코드는 각 스레드가 독립적으로 실행
    #pragma omp parallel
    {
        // 각 스레드가 가지는 private 변수들
        int i, uniqueID, x, nthrds, sum;

        // 현재 스레드의 고유 번호 (0부터 시작)
        uniqueID = omp_get_thread_num();

        // 전체 스레드 개수
        nthrds = omp_get_num_threads();

        sum = 0;

        // 마스터 스레드(0번)만 전체 스레드 개수 저장
        if (uniqueID == 0) {
            nthreads = nthrds;
            printf("총 스레드 개수: %d\n", nthreads);
        }

        // === 핵심 병렬 계산 ===
        // 각 스레드가 자신의 번호부터 시작해서
        // 스레드 개수만큼 건너뛰며 계산
        // 예: 스레드 0 → 0, 4, 8, 12, ...
        //     스레드 1 → 1, 5, 9, 13, ...
        //     스레드 2 → 2, 6, 10, 14, ...
        //     스레드 3 → 3, 7, 11, 15, ...
        for (i = uniqueID; i < num_steps; i += nthrds)
        {
            x = i + constant;
            sum += x;
        }

        // 각 스레드의 부분 결과를 합산
        #pragma omp critical
        {
            total += sum * constant;
        }

        // 병렬 영역 종료 - 암묵적 barrier (모든 스레드 동기화)
    }

    // 병렬 실행 종료 시간 측정
    double end_par = omp_get_wtime();
    double time_par = end_par - start_par;

    printf("\n결과: %d\n", total);
    printf("실행 시간: %.6f 초\n\n", time_par);


    // === 성능 비교 ===
    printf("=== 성능 비교 ===\n");
    printf("순차 실행 시간: %.6f 초\n", time_seq);
    printf("병렬 실행 시간: %.6f 초\n", time_par);
    printf("속도 향상: %.2fx 배\n", time_seq / time_par);
    printf("효율성: %.1f%% (이상적: 100%% × 스레드 수)\n",
           (time_seq / time_par) / 4 * 100);

    // 결과 검증
    if (total_seq == total) {
        printf("\n✓ 결과 검증 성공: 순차와 병렬 결과 일치\n");
    } else {
        printf("\n✗ 결과 검증 실패: 순차(%d) ≠ 병렬(%d)\n",
               total_seq, total);
    }

    return 0;
}
