################################################################################
# CUDA Makefile – wersja po refaktoryzacji (main.cpp + CudaCompute.cu)
################################################################################

CUDA_PATH ?= /usr/local/cuda

# ------------------------------------------------------------------------------
# Architektura i system
# ------------------------------------------------------------------------------
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
TARGET_SIZE := $(shell getconf LONG_BIT)

HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_OS ?= $(HOST_OS)

# ------------------------------------------------------------------------------
# Kompilatory
# ------------------------------------------------------------------------------
HOST_COMPILER ?= g++
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# ------------------------------------------------------------------------------
# Pliki projektu
# ------------------------------------------------------------------------------
CU_SRC   := CudaCompute.cu
CPP_SRC  := main.cpp FileManager.cpp

CU_OBJ   := $(CU_SRC:.cu=.o)
CPP_OBJ  := $(CPP_SRC:.cpp=.o)

OBJ      := $(CU_OBJ) $(CPP_OBJ)

TARGET   := kernel

# ------------------------------------------------------------------------------
# Flagi
# ------------------------------------------------------------------------------
NVCCFLAGS := -m$(TARGET_SIZE)
CCFLAGS   :=
LDFLAGS   :=

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += -Xcompiler -Wall
ALL_CCFLAGS += --std=c++11

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)

# ------------------------------------------------------------------------------
# Gencode (architektury GPU)
# ------------------------------------------------------------------------------
SMS ?= 50 52 60 61 70 75 80 86

GENCODE_FLAGS :=
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

HIGHEST_SM := $(lastword $(sort $(SMS)))
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)

# ------------------------------------------------------------------------------
# Reguły
# ------------------------------------------------------------------------------
all: $(TARGET)

# kompilacja CUDA
%.o: %.cu
	$(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $< -o $@

# kompilacja C++
%.o: %.cpp
	$(NVCC) $(ALL_CCFLAGS) -c $< -o $@

# linkowanie (MUSI być nvcc)
$(TARGET): $(OBJ)
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $^

# ------------------------------------------------------------------------------
# Run / Clean
# ------------------------------------------------------------------------------
run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o

clobber: clean
