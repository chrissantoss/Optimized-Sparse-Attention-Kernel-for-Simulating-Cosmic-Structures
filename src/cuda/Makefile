CUDA_PATH ?= /usr/local/cuda
CUTLASS_PATH ?= $(HOME)/cutlass

CXX := g++
NVCC := $(CUDA_PATH)/bin/nvcc

# Compiler flags
CXXFLAGS := -std=c++14 -O3 -Wall -Wextra -fPIC
NVCCFLAGS := -std=c++14 -O3 -Xcompiler -fPIC

# Include paths
INCLUDES := -I$(CUDA_PATH)/include -I$(CUTLASS_PATH)/include -I$(CUTLASS_PATH)/tools/util/include

# Library paths
LDFLAGS := -L$(CUDA_PATH)/lib64
LDLIBS := -lcudart -lcublas

# Source files
CUDA_SRCS := sparse_attention.cu
CUDA_OBJS := $(CUDA_SRCS:.cu=.o)

# Target library
TARGET := libsparse_attention.so

# Debug flags
ifdef DEBUG
    CXXFLAGS += -g -DDEBUG
    NVCCFLAGS += -g -G -DDEBUG
endif

# Detect GPU architecture
ARCH := $(shell $(NVCC) --help | grep -o "sm_[0-9][0-9]" | sort -r | head -1)
ifeq ($(ARCH),)
    ARCH := sm_70  # Default to Volta if detection fails
endif
NVCCFLAGS += -arch=$(ARCH)

# Check for Tensor Cores
TENSOR_CORES := $(shell echo $(ARCH) | awk '{ if ($$1 >= "sm_70") print "1"; else print "0"; }')
ifeq ($(TENSOR_CORES),1)
    NVCCFLAGS += -DUSE_TENSOR_CORES
endif

# Rules
.PHONY: all clean debug

all: $(TARGET)

debug:
	$(MAKE) DEBUG=1

$(TARGET): $(CUDA_OBJS)
	$(CXX) -shared -o $@ $^ $(LDFLAGS) $(LDLIBS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(CUDA_OBJS) $(TARGET)

# Print debug info
info:
	@echo "CUDA_PATH: $(CUDA_PATH)"
	@echo "CUTLASS_PATH: $(CUTLASS_PATH)"
	@echo "GPU Architecture: $(ARCH)"
	@echo "Tensor Cores: $(TENSOR_CORES)"
	@echo "NVCC Version: $$($(NVCC) --version | grep release | awk '{print $$5}')"
	@echo "CUDA Includes: $(INCLUDES)" 