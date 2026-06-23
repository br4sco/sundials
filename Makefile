BUILD_DIR := build-dir

.PHONY: all
all: build

.PHONY: configure
configure: $(BUILD_DIR)

build: $(BUILD_DIR)
	make -C $(BUILD_DIR)

test: $(BUILD_DIR)
	cd $(BUILD_DIR) &&\
	make CFLAGS=-Wpedantic && make CTEST_OUTPUT_ON_FAILURE=1 test

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) &&\
	cmake\
		-DCMAKE_BUILD_TYPE=Debug\
		-DCMAKE_C_FLAGS=-Wpedantic\
		-DBUILD_ARKODE=0\
		-DBUILD_KINSOL=0\
		-DBUILD_CVODE=0\
		-DBUILD_CVODES=0\
		-DBUILD_IDA=0\
		-DENABLE_KLU=1\
		-DKLU_LIBRARY_DIR=$(KLU_LIBRARY_DIR)\
		-DKLU_INCLUDE_DIR=$(KLU_INCLUDE_DIR)\
		-DSUNDIALS_ENABLE_EXTERNAL_ADDONS=1\
		-DSUNDIALS_TEST_UNITTESTS=1\
		-DCMAKE_INSTALL_PREFIX=/tmp/sundials\
		-DSUNDIALS_EXAMPLES_INSTALL_PATH=/tmp/sundials\
		-DCMAKE_EXPORT_COMPILE_COMMANDS=1\
		..

install: $(BUILD_DIR)
	make -C $(BUILD_DIR) install

.PHONY: format
format:
	clang-format -i $(shell find external/dd/src -name "*.h" -o -name "*.c")

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
