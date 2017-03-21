#-------------------------------------------------
#
# Project created by QtCreator 2017-03-19T07:45:02
#
#-------------------------------------------------

QT += core gui opengl

greaterThan(QT_MAJOR_VERSION, 5): QT += widgets

TARGET = LumenEngine
TEMPLATE = app


SOURCES += main.cpp \
        src/interface/GUI.cpp \
    src/scene/sceneParser.cpp

HEADERS += \
        src/cuda/renderKernel.h \
        src/interface/GUI.h \
    src/scene/sceneParser.h

LIBS += -lGL -lGLEW -lGLU -lglut

QMAKE_CXXFLAGS += -std=c++11 -O3


## CUDA setup ##

GENCODE = arch=compute_52,code=sm_52

DEFINES += NOMINMAX

CUDA_SOURCES = "$$PWD"/src/cuda/renderKernel.cu

SOURCES += src/cuda/renderKernel.cu
SOURCES -= src/cuda/renderKernel.cu

CUDA_DIR = /usr/local/cuda
CUDA_SDK = /usr/local/cuda/samples

INCLUDEPATH += $$CUDA_DIR/include

QMAKE_LIBDIR += $$CUDA_DIR/lib64
LIBS += -lcudart -lcudadevrt

CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

NVCCFLAGS = --compiler-options  -fno-strict-aliasing --ptxas-options=-v -maxrregcount 20 --use_fast_math

cudaIntr.input = CUDA_SOURCES
cudaIntr.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}.o

cudaIntr.commands = $$CUDA_DIR/bin/nvcc -m64 -g -gencode $$GENCODE -dc $$NVCCFLAGS $$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
cudaIntr.variable_out = CUDA_OBJ
cudaIntr.variable_out += OBJECTS
cudaIntr.clean = cudaIntrObj/*.o

QMAKE_EXTRA_UNIX_COMPILERS += cudaIntr

cuda.input = CUDA_OBJ
cuda.output = ${QMAKE_FILE_BASE}_link.o

cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -g -gencode $$GENCODE  -dlink    ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}

QMAKE_EXTRA_UNIX_COMPILERS += cuda
