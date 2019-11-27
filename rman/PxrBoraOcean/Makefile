include ./Makefile.base

PROJECT_NAME := PxrBoraOcean
TARGET_PATH := .

#-------------------------------------------------------Base
INCLUDES := -I/usr/local/include
#INCLUDES += -I/usr/local/include/eigen3
INCLUDES += -I/usr/local/cuda/include
INCLUDES += -I/netapp/backstage/pub/lib/bora/header
INCLUDES += -I/netapp/backstage/pub/lib/extern/include/bullet
INCLUDES += -I$(EXTERN_LIBRARY_PATH)/include

#LINKS := -L/netapp/backstage/pub/lib/bora/lib
LINKS := -L../../build/lib
LINKS += -lBoraBase

#-------------------------------------------------------Prman
INCLUDES	+= -I$(PRMAN_INSTALL_PATH)/include
LINKS		+= -L$(PRMAN_INSTALL_PATH)/lib
LINKS		+= -lprman-$(PRMAN_VER)

INCLUDES += -I$(HEADER_PATH) -I$(SOURCE_PATH)
INCLUDES += -I../../base/header



# CCFLAGS := -O3 -m64 -fpic -fopenmp -std=c++11
LDFLAGS := -shared
CCFLAGS := -fPIC -std=c++11

DEFINES := -DGLM_ENABLE_EXPERIMENTAL

.SILENT:

all: $(PROJECT_NAME)
	@echo -e '\e[1;33m$(PROJECT_NAME)::build complete.\e[m'

# .o -> .so
$(PROJECT_NAME): $(OBJ_LIST)
	@echo -e '\e[1;35mBuilding : $@\e[m'
	@echo -e '\e[1;32m ----- $$MAYA_INSTALL_PATH=$(MAYA_INSTALL_PATH)\e[m'
	@echo -e '\e[1;32m ----- $$PRMAN_INSTALL_PATH=$(PRMAN_INSTALL_PATH)\e[m'
	@echo -e '\e[1;32m ----- $$EXTERN_LIBRARY_PATH=$(EXTERN_LIBRARY_PATH)\e[m'
	@echo -e '\e[1;32m ----- $$ZELOS_PATH=$(ZELOS_PATH)\e[m'

	$(CC) $(LDFLAGS) $(LINKS) $(RPATHS) -o $(TARGET_PATH)/$@.so $(OBJ_LIST)

	#mkdir -p -m 755 ./$(TARGET_PATH)/Args
	#cp -f $(SOURCE_PATH)/$(PROJECT_NAME).a ./$(TARGET_PATH)/Args/$(PROJECT_NAME).args

# .cpp -> .o
$(OBJECT_PATH)/%.o: $(SOURCE_PATH)/%.cpp
	$(CC) $(CCFLAGS) $(INCLUDES) $(DEFINES) -c $< -o $@

clean:
	rm -vf $(OBJECT_PATH)/*.o
	rm -vf $(TARGET_PATH)/$(PROJECT_NAME).so
	rm -vf $(TARGET_PATH)/Args/$(PROJECT_NAME).args
