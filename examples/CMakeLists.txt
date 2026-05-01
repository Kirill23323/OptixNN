add_executable(build_index build_index.cpp)
target_link_libraries(build_index PRIVATE OptixNN)

add_executable(search_example search_example.cpp)
target_link_libraries(search_example PRIVATE OptixNN)

add_executable(omp_example omp_example.cpp)
target_link_libraries(omp_example PRIVATE OptixNN)
