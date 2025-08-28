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
    "dd_pivot_test_linsys_1\;"
    "dd_pivot_test_nonlinsys_1\;"
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
    "model_pendulum_ASA\;")

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

message(STATUS "Added DD C tests ")
