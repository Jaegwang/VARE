##----------##
## Makefile ##
##-------------------------------------------------------##
## author: Jaegwang Lim @ Dexter Studios                 ##
## last update: 2018.01.16                               ##
##-------------------------------------------------------##

include ../Makefile.global

CC := g++
CXXFLAGS := -Wall -W -std=c++11 -m64 -fPIC -O2 -fopenmp

INCDIRS := -I.
INCDIRS += -I./header
INCDIRS += -I/usr/local/inlcude
INCDIRS += -I/usr/local/cuda/include
INCDIRS += -I/usr/local/include/eigen3
INCDIRS += -I../base/header
INCDIRS += -I$(MAYA_INSTALL_PATH)/include
INCDIRS += -I$(HOU_INSTALL_PATH)/toolkit/include
INCDIRS += -I$(RMAN_INSTALL_PATH)/include

LIBS := -L.
LIBS += -L../build/base -lBoraBase
LIBS += -L/usr/local/cuda/lib64
LIBS += -L/usr/local/lib
LIBS += -L$(MAYA_INSTALL_PATH)/lib
LIBS += -lOpenMaya -lOpenMayaAnim -lFoundation -lOpenMayaUI -lOpenMayaFX
LIBS += -L$(HOU_INSTALL_PATH)/dsolib
LIBS += -lHoudiniGEO -lHoudiniOP1 -lHoudiniOP2 -lHoudiniOP3 -lHoudiniOPZ -lHoudiniPRM
LIBS += -lHoudiniRAY -lHoudiniSIM -lHoudiniUI -lHoudiniUT

DEFINES := $(HDEFINES)
DEFINES += -DGLM_FORCE_CUDA 
DEFINES += -DGLM_ENABLE_EXPERIMENTAL

BUILD_PATH   := ../build/maya
PROJECT_NAME := BoraForMaya

HEADER_PATH := header
SOURCE_PATH := source
OBJECT_PATH := object

CPP_SOURCES := $(wildcard $(SOURCE_PATH)/*.cpp)
HDR_SOURCES := $(wildcard $(HEADER_PATH)/*.h)
OBJ_LIST    := $(patsubst $(SOURCE_PATH)/%.cpp, $(OBJECT_PATH)/%.o, $(CPP_SOURCES) )

.SILENT:

all: comm $(OBJECT_PATH)/$(PROJECT_NAME).so

comm:
	@echo -e '\e[1;32m Building $(PROJECT_NAME)... \e[m'
	mkdir -p -m 755 $(BUILD_PATH)/$(MAYA_VER)/scripts
	cp -rf script/*.mel $(BUILD_PATH)/$(MAYA_VER)/scripts

# .o -> .so
$(OBJECT_PATH)/$(PROJECT_NAME).so: $(OBJ_LIST)
	$(CC) $(CXXFLAGS) $(DEFINES) $(LIBS) $(SWITCHES) -shared $(OBJ_LIST) -o $@
	mkdir -p -m 755 $(BUILD_PATH)/$(MAYA_VER)/plug-ins
	cp -rf $@ $(BUILD_PATH)/$(MAYA_VER)/plug-ins 

# .cpp -> .o
$(OBJECT_PATH)/%.o: $(SOURCE_PATH)/%.cpp
	$(CC) $(CXXFLAGS) $(DEFINES) $(INCDIRS) $(SWITCHES) -c $< -o $@

maya:
	./maya.bash $(MAYA_VER) $(RMAN_VER) $(HOU_INSTALL_PATH)

clean:
	rm -rf $(OBJECT_PATH)/*
	rm -rf $(BUILD_PATH)/*	

