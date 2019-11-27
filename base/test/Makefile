HOME_DIR   := $(shell pwd)
SOURCES    := $(wildcard *.cpp)
SOURCESCPP := $(notdir $(SOURCES))
PROJECTS   := $(SOURCESCPP:.cpp=)

SOURCES    := $(wildcard *.cpp)
SOURCESCPP := $(notdir $(SOURCES))
ALL_PROJ   := $(SOURCESCPP:.cpp=)

CC         := g++
CCFLAGS    := -O3 -m64 -std=c++11
INCLUDES   := -I.
INCLUDES   += -I../header
INCLUDES   += -I/usr/local/include
INCLUDES   += -I/usr/local/include/eigen3
INCLUDES   += -I/usr/local/cuda/include
LINKS      := -L/usr/local/lib
LINKS      += -L../../build/lib
LINKS      += -lhdf5 -lAlembic
LINKS      += -L/usr/local/cuda/lib64
LINKS      += -lHalf -ltbb -lcudart -lcublas -lopenvdb
LINKS      += -L/usr/lib64 -lglut -lGLU -lGL -lGLEW
LINKS      += -L../../build/lib -lBoraBase
DEFINES    := -DGLM_FORCE_CUDA 
DEFINES    += -DGLM_ENABLE_EXPERIMENTAL

RPATHS     := -Wl,-rpath=../../build/lib

.SILENT:

all: $(PROJECTS)

%: %.cpp
	@echo 'Compiling: $(notdir $<) -> $(notdir $@)'
	$(CC) $(CCFLAGS) $(DEFINES) $(RPATHS) -o $@ $< $(INCLUDES) $(LINKS)

clean:
	rm -rf $(PROJECTS) result.*

