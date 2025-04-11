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
		-DSUNDIALS_ENABLE_EXTERNAL_ADDONS=1\
		-DSUNDIALS_TEST_UNITTESTS=1\
		-DCMAKE_INSTALL_PREFIX=/tmp/sundials\
		-DSUNDIALS_EXAMPLES_INSTALL_PATH=/tmp/sundials\
		-DCMAKE_EXPORT_COMPILE_COMMANDS=1\
		..

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
