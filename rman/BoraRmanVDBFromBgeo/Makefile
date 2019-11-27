##----------##
## Makefile ##
##-------------------------------------------------------##
## author: Jaegwang Lim @ Dexter Studios                 ##
## last update: 2017.11.13                               ##
##-------------------------------------------------------##

include ../../Makefile.global

CC := g++
CCFLAGS := -Wall -W -O3 -m64 -fPIC -fopenmp -std=c++11

PROJECT_NAME := BoraRmanVDBFromPoints
BUILD_PATH   := ../../build/rman

INCLUDES := -I.
INCLUDES += -I/usr/local/include
INCLUDES += -I$(RMAN_INSTALL_PATH)/include
INCLUDES += -I$(HOU_INSTALL_PATH)/toolkit/include
INCLUDES += -I/usr/local/cuda/include
INCLUDES += -I/usr/local/include/eigen3
INCLUDES += -I/usr/local/include/bullet
INCLUDES += -I../../base/header

LINKS := -L.
LINKS += -L$(HOU_INSTALL_PATH)/dsolib
LINKS += -lopenvdb_sesi
LINKS += -L../../build/lib
LINKS += -lBoraBase

RPATHS := -Wl,-rpath=.
RPATHS += -Wl,-rpath=/home/jaegwang.lim/work/Bora/build/lib
RPATHS += -Wl,-rpath=/lib64
RPATHS += -Wl,-rpath=/usr/local/lib
RPATHS += -Wl,-rpath=$(HOU_INSTALL_PATH)/dsolib

DEFINES := $(HDEFINES)
DEFINES += -DGLM_FORCE_CUDA 
DEFINES += -DGLM_ENABLE_EXPERIMENTAL

CPP_SOURCES := $(wildcard *.cpp)

.SILENT:

all: $(BUILD_PATH)/$(PROJECT_NAME).so

# .cpp -> .so
$(BUILD_PATH)/$(PROJECT_NAME).so: $(CPP_SOURCES)
	$(CC) $(CCFLAGS) $(INCLUDES) $(DEFINES) $(LINKS) $(RPATHS) $(SWITCHES) -shared $(CPP_SOURCES) -o $@

clean:
	rm -f $(BUILD_PATH)/$(PROJECT_NAME).so

