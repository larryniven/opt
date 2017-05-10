CXXFLAGS += -std=c++11 -I ../
NVCCFLAGS += -std=c++11 -I ../
AR = gcc-ar

obj = opt.o

.PHONY: all clean gpu

all: libopt.a

clean:
	-rm *.o
	-rm libopt.a

gpu: liboptgpu.a

libopt.a: $(obj)
	$(AR) rcs $@ $^

liboptgpu.a: $(obj) opt-gpu.o
	$(AR) rcs $@ $^

opt-gpu.o: opt-gpu.cu
	nvcc $(NVCCFLAGS) -c opt-gpu.cu
