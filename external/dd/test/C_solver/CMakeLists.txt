# List of test tuples of the form "name\;args"
set(unit_tests
    "dd_alg_test_lexiographical_combination_1_over_1\;"
    "dd_alg_test_lexiographical_combination_4_over_2\;"
    "dd_darr_test_create_destroy\;"
    "dd_darr_push\;"
    "dd_darr_set_get\;"
    "dd_matrix_test_dense_create_destroy\;"
    "dd_matrix_test_dense_create_destroy_workspace\;"
    "dd_matrix_test_dense_pivot\;"
    "dd_matrix_test_dense_pivot_example_1_1\;"
    "dd_matrix_test_dense_pivot_example_1_2\;"
    "dd_matrix_test_dense_clonesub\;"
    "dd_matrix_test_dense_copysub\;"
    "dd_structure_test_create_destroy\;"
    "dd_structure_test_lotka_volterra\;"
    "dd_structure_test_pendulum\;"
    "dd_pivot_test_create_destroy\;"
    "dd_pivot_test_pendulum_1\;"
    "dd_pivot_test_pendulum_2\;"
    "dd_pivot_test_pendulum_3\;"
    "dd_pivot_test_linsys_1\;"
    "dd_pivot_test_nonlinsys_1\;"
    "dd_state_pivot_test\;"
    "model_lotka-volterra\;n"
    "model_lotka-volterra\;d"
    "model_lotka-volterra\;r"
    "model_lotka-volterra\;c"
    "model_pendulum\;n"
    "model_pendulum\;d"
    "model_pendulum\;r"
    "model_pendulum\;c"
    "model_pendulum_FSA\;"
    "model_pendulum_FSA\;analytic_sens_residual"
    "model_pendulum_ASA\;"
    "dd_test_reinitb_interior\;"
    "dd_test_reinitb_after_full_pass\;h n"
    "dd_test_reinitb_after_full_pass\;h d"
    "dd_test_reinitb_after_full_pass\;h r"
    "dd_test_reinitb_after_full_pass\;h c"
    "dd_test_reinitb_after_full_pass\;p n"
    "dd_test_reinitb_after_full_pass\;p d"
    "dd_test_reinitb_after_full_pass\;p r"
    "dd_test_reinitb_after_full_pass\;p c"
    "dd_test_reinitb_second_pass\;h n"
    "dd_test_reinitb_second_pass\;h d"
    "dd_test_reinitb_second_pass\;h r"
    "dd_test_reinitb_second_pass\;h c"
    "dd_test_reinitb_second_pass\;p n"
    "dd_test_reinitb_second_pass\;p d"
    "dd_test_reinitb_second_pass\;p r"
    "dd_test_reinitb_second_pass\;p c")

# Sundials KLU linear solver module
set(SUNLINSOLKLU_LIBS sundials_sunlinsolklu)

# KLU libraries
list(APPEND SUNLINSOLKLU_LIBS)

# Add the build and install targets for each test
foreach(test_tuple ${unit_tests})

  # parse the test tuple
  list(GET test_tuple 0 test)
  list(GET test_tuple 1 test_args)

  # check if this test has already been added, only need to add test source
  # files once for testing with different inputs
  if(NOT TARGET ${test})

    # test source files
    add_executable(${test} ${test}.c)

    set_target_properties(${test} PROPERTIES FOLDER "dd_tests")

    # include location of public and private header files
    target_include_directories(
      ${test}
      PRIVATE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
              ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src
              ${CMAKE_SOURCE_DIR}/external/dd/src)

    # libraries to link against
    target_link_libraries(${test} sundials_idas sundials_nvecserial dd
                          ${EXE_EXTRA_LINK_LIBS} ${SUNLINSOLKLU_LIBS})

  endif()

  # check if test args are provided and set the test name
  if("${test_args}" STREQUAL "")
    set(test_name ${test})
  else()
    string(REPLACE " " "_" test_name "${test}_${test_args}")
    string(REPLACE " " ";" test_args "${test_args}")
  endif()

  # add test to regression tests
  add_test(NAME ${test_name} COMMAND ${test} ${test_args})

endforeach()

# Exploratory driver (not a regression test): sweeps the time between pivot
# checks and logs IDAPrintAllStats at each successful pivot. Build only, run
# manually with varying dt_check, e.g.: ./model_pendulum_pivot_interval d 0.5
add_executable(model_pendulum_pivot_interval model_pendulum_pivot_interval.c)

set_target_properties(model_pendulum_pivot_interval PROPERTIES FOLDER
                                                               "dd_tests")

target_include_directories(
  model_pendulum_pivot_interval
  PRIVATE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
          ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src
          ${CMAKE_SOURCE_DIR}/external/dd/src)

target_link_libraries(
  model_pendulum_pivot_interval sundials_idas sundials_nvecserial dd
  ${EXE_EXTRA_LINK_LIBS} ${SUNLINSOLKLU_LIBS})

# Exploratory driver (not a regression test): no pivot-check schedule at all --
# DDSPPivotStaged is only called reactively, when IDA's failure counters grow.
# Build only, run manually, e.g.: ./model_pendulum_pivot_adaptive d
add_executable(model_pendulum_pivot_adaptive model_pendulum_pivot_adaptive.c)

set_target_properties(model_pendulum_pivot_adaptive PROPERTIES FOLDER
                                                               "dd_tests")

target_include_directories(
  model_pendulum_pivot_adaptive
  PRIVATE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
          ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src
          ${CMAKE_SOURCE_DIR}/external/dd/src)

target_link_libraries(
  model_pendulum_pivot_adaptive sundials_idas sundials_nvecserial dd
  ${EXE_EXTRA_LINK_LIBS} ${SUNLINSOLKLU_LIBS})

# Exploratory driver (not a regression test): same as
# model_pendulum_pivot_adaptive but tracks only IDA's error test fail counter
# (netf) -- instrumenting all three failure counters showed netf alone accounted
# for 45/46 successful pivots, so ncfn/nnf are dropped here. Build only, run
# manually, e.g.: ./model_pendulum_pivot_adaptive_netf d
add_executable(model_pendulum_pivot_adaptive_netf
               model_pendulum_pivot_adaptive_netf.c)

set_target_properties(model_pendulum_pivot_adaptive_netf PROPERTIES FOLDER
                                                                    "dd_tests")

target_include_directories(
  model_pendulum_pivot_adaptive_netf
  PRIVATE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
          ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src
          ${CMAKE_SOURCE_DIR}/external/dd/src)

target_link_libraries(
  model_pendulum_pivot_adaptive_netf sundials_idas sundials_nvecserial dd
  ${EXE_EXTRA_LINK_LIBS} ${SUNLINSOLKLU_LIBS})

# Exploratory driver (not a regression test): same as
# model_pendulum_pivot_adaptive_netf but suppresses further checks for dt time
# units after a successful pivot, to cut down on false-positive checks that tend
# to cluster right after a real pivot. Build only, run manually with varying dt,
# e.g.: ./model_pendulum_pivot_adaptive_netf_cooldown d 0.5
add_executable(model_pendulum_pivot_adaptive_netf_cooldown
               model_pendulum_pivot_adaptive_netf_cooldown.c)

set_target_properties(model_pendulum_pivot_adaptive_netf_cooldown
                      PROPERTIES FOLDER "dd_tests")

target_include_directories(
  model_pendulum_pivot_adaptive_netf_cooldown
  PRIVATE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
          ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src
          ${CMAKE_SOURCE_DIR}/external/dd/src)

target_link_libraries(
  model_pendulum_pivot_adaptive_netf_cooldown sundials_idas sundials_nvecserial
  dd ${EXE_EXTRA_LINK_LIBS} ${SUNLINSOLKLU_LIBS})

# Benchmark variants of the three pivoting-approach drivers above, for comparing
# their raw overhead with hyperfine. Same source, but compiled with DD_BENCH
# defined so DD_LOG/DD_PRINT_ALL_STATS compile out to nothing -- isolating the
# cost of the pivoting logic (DDSPPivotStaged calls, IDAGetNum* queries,
# DDSolve) from file I/O, which would otherwise dominate the timing and mask any
# difference between the three approaches.
function(dd_add_bench_target base_target source)
  set(bench_target ${base_target}_bench)
  add_executable(${bench_target} ${source})
  target_compile_definitions(${bench_target} PRIVATE DD_BENCH)
  set_target_properties(${bench_target} PROPERTIES FOLDER "dd_tests")
  target_include_directories(
    ${bench_target}
    PRIVATE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
            ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/external/dd/src)
  target_link_libraries(${bench_target} sundials_idas sundials_nvecserial dd
                        ${EXE_EXTRA_LINK_LIBS} ${SUNLINSOLKLU_LIBS})
endfunction()

dd_add_bench_target(model_pendulum_pivot_interval
                    model_pendulum_pivot_interval.c)
dd_add_bench_target(model_pendulum_pivot_adaptive_netf
                    model_pendulum_pivot_adaptive_netf.c)
dd_add_bench_target(model_pendulum_pivot_adaptive_netf_cooldown
                    model_pendulum_pivot_adaptive_netf_cooldown.c)

message(STATUS "Added DD C tests ")
