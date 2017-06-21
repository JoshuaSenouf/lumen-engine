#ifndef RENDERER_H
#define RENDERER_H

#include <iostream>
#include <fstream>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cuda_runtime.h>
#include "cuda_gl_interop.h"

#include "renderKernel.h"
#include "gui.h"
#include "camera.h"
#include "scene.h"


class Renderer
{
    public:
        Renderer();
        int runRenderer();
        void initCUDAData();
        void cleanCUDAData();
        void initRenderVBO(GLuint* renderVBO, cudaGraphicsResource** cudaGRBuffer, unsigned int cudaFlags);
        void cleanRenderVBO(GLuint* renderVBO, cudaGraphicsResource* cudaGRBuffer);
        void initCUDAScene();
        void cleanCUDAScene();
        void resetRender();
        void displayGLBuffer();

        void keyboardCallback(ImGuiIO* guiIO);
        void mouseCallback(ImGuiIO* guiIO, float mousePosX, float mousePosY);

    private:
        bool firstMouse = true;
        bool guiIsOpen = true;
        bool renderReset = false;

        int frameCounter = 0;
        int sphereCount;

        GLuint renderWidth = 1280;
        GLuint renderHeight = 720;
        GLuint renderSamples = 4;
        GLuint renderBounces = 4;
        GLuint renderVBO;

        GLfloat deltaTime = 0.0f;
        GLfloat lastFrame = 0.0f;
        GLfloat lastPosX = renderWidth / 2;
        GLfloat lastPosY = renderHeight / 2;

        glm::vec3* outputBuffer;
        glm::vec3* accumulationBuffer;

        SphereObject* spheresList;

        cudaGraphicsResource* cudaGRBuffer;

        cudaStream_t cudaDataStream;

        GLFWwindow* window;

        GUI lumenGUI;

        Camera cudaCamera;
        CameraInfo* cudaCameraInfo;
};


#endif // RENDERER_H
