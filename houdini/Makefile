##----------##
## Makefile ##
##-------------------------------------------------------##
## author: Jaegwang Lim @ Dexter Studios                 ##
## last update: 2017.09.15                               ##
##-------------------------------------------------------##

include ../Makefile.global

CC := g++
CXXFLAGS := -Wall -W -std=c++11 -m64 -fPIC -O2 -fopenmp -fno-strict-aliasing

HFS := $(HOU_INSTALL_PATH)

DEFINES := $(HDEFINES)
DEFINES += -DGLM_FORCE_CUDA 
DEFINES += -DGLM_ENABLE_EXPERIMENTAL

INCDIRS := -I.
INCDIRS += -isystem $(HFS)/toolkit/include
INCDIRS += -I./header -I/usr/local/include
INCDIRS += -I/usr/local/include/eigen3
INCDIRS += -I../base/header
INCDIRS += -I/usr/local/include/boost-1_55
INCDIRS += -I/usr/local/cuda/include
INCDIRS += -I/usr/include/python2.7

LIBS    := -L.
LIBS    += -L$(HFS)/dsolib
LIBS    += -L./lib -L/usr/local/lib
LIBS    += -lGL -lGLU -lGLEW -lX11 -lXext -lXi -ldl -lpthread
LIBS    += -L../build/lib -lBoraBase

BUILD_PATH   := ../build/houdini
PROJECT_NAME := BoraForHoudini

HEADER_PATH := header
SOURCE_PATH := source
OBJECT_PATH := object

CPP_SOURCES := $(wildcard $(SOURCE_PATH)/*.cpp)
HDR_SOURCES := $(wildcard $(HEADER_PATH)/*.h)
OBJ_LIST    := $(patsubst $(SOURCE_PATH)/%.cpp, $(OBJECT_PATH)/%.o, $(CPP_SOURCES) )

.SILENT:

all: pre $(OBJECT_PATH)/$(PROJECT_NAME).so

pre:
	@echo -e '\e[1;32m Building $(PROJECT_NAME)... \e[m'
	chmod 755 header/*
	chmod 755 source/*
	chmod 755 otls/*
	mkdir -p -m 755 $(BUILD_PATH)/otls
	cp -rf otls/*.hda $(BUILD_PATH)/otls
	cp -rf otls/*.otl $(BUILD_PATH)/otls
	
# .o --> .so
$(OBJECT_PATH)/$(PROJECT_NAME).so: $(OBJ_LIST) $(HDR_SOURCES)
	$(CC) $(LIBS) -shared $(OBJ_LIST) -o $@
	mkdir -p -m 755 $(BUILD_PATH)/plugins
	cp -rf $@ $(BUILD_PATH)/plugins

# .cpp --> .o
$(OBJECT_PATH)/%.o: $(SOURCE_PATH)/%.cpp $(HDR_SOURCES)
	mkdir -p -m 755 $(OBJECT_PATH)
	$(CC) $(CXXFLAGS) $(INCDIRS) $(DEFINES) $(SWITCHES) -c $< -o $@

clean:
	rm -rf $(OBJECT_PATH)/*
	rm -rf $(BUILD_PATH)/*

hou:
	./houdini.bash $(HFS)

