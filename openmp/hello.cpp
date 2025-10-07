#include <bits/stdc++.h>
#include <omp.h>

int main() {
    int thread_counter = 0;
#pragma omp parallel
    std::cout << "hello world" << thread_counter << "\n";
    return 0;
}

