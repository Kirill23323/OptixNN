#include <iostream>
#include <omp.h>

int main() {
    std::cout << "OpenMP Test Program" << std::endl;
    
#ifdef _OPENMP
    std::cout << "OpenMP is enabled!" << std::endl;
    std::cout << "Number of threads available: " << omp_get_max_threads() << std::endl;
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        
        #pragma omp critical
        {
            std::cout << "Hello from thread " << thread_id 
                      << " of " << num_threads << std::endl;
        }
    }
#else
    std::cout << "OpenMP is NOT enabled!" << std::endl;
    std::cout << "Running in serial mode." << std::endl;
#endif

    return 0;
}