EBT = ../ebt

CXXFLAGS += -std=c++11 -I $(EBT)
AR = gcc-ar

obj = opt.o

.PHONY: all clean

all: libopt.a

clean:
	-rm *.o
	-rm libopt.a

libopt.a: $(obj)
	$(AR) rcs $@ $^
