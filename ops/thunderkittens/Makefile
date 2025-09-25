# Compiler
NVCC?=nvcc
GPU=Hopper

TARGET=flash_ffn_moe
SRC=flash_ffn_moe.cu

TORCH_INC      := $(shell python3 -c "import torch, os; base=os.path.dirname(torch.__file__); print(base + '/include')")
TORCH_INC_API  := $(shell python3 -c "import torch, os; base=os.path.dirname(torch.__file__); print(base + '/include/torch/csrc/api/include')")
TORCH_LIB      := $(shell python3 -c "import torch, os; base=os.path.dirname(torch.__file__); print(base + '/lib')")
GLIBCXX_ABI := $(shell python3 -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

NVCCFLAGS := -DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -x cu -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart -lcublas
NVCCFLAGS += -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype $(shell python3 -m pybind11 --includes) $(shell python3-config --ldflags) -shared -fPIC
NVCCFLAGS += -I$(TORCH_INC) -I$(TORCH_INC_API) -L$(TORCH_LIB) -Wl,-rpath,$(TORCH_LIB)
NVCCFLAGS += -ltorch -ltorch_python -lc10
NVCCFLAGS += -D_GLIBCXX_USE_CXX11_ABI=$(GLIBCXX_ABI)
NVCCFLAGS += -diag-suppress 3189
NVCCFLAGS += -Xptxas -maxrregcount=192 --relocatable-device-code=false

# Conditional setup based on the target GPU
ifeq ($(GPU),Ada)
NVCCFLAGS += -DKITTENS_4090 -arch=sm_89
else ifeq ($(GPU),Ampere)
NVCCFLAGS += -DKITTENS_A100 -arch=sm_86
else ifeq ($(GPU),Hopper)
NVCCFLAGS += -DKITTENS_HOPPER -arch=sm_90a
else
$(error Invalid GPU Type: $(GPU), only use Ada, Ampere, or Hopper. Falling back to Ampere.)
NVCCFLAGS += -DKITTENS_A100 -arch=sm_80
endif

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)$(shell python3-config --extension-suffix)

# Clean target
clean:
	rm -f $(TARGET) 