#include <bits/stdc++.h>
#include <omp.h>

static long num_steps = 1000000;

int main() {
    // === 버그 있는 코드 (Race Condition) ===
    int total_bug = 0;

    #pragma omp parallel num_threads(4)
    {
        int i, x;
        int uniqueID = omp_get_thread_num();
        int nthrds = omp_get_num_threads();

        for (i = uniqueID; i < num_steps; i += nthrds) {
            x = i + 12;
            total_bug += x;
        }
    }

    printf("total_bug: %d\n", total_bug);



}
