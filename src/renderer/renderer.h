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

#include "cuda_gl_interop.h"

#include "renderKernel.h"
#include "gui.h"
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
		void displayGLBuffer();

	private:
		bool firstMouse = true;
		bool guiIsOpen = true;
		bool keys[1024];

		int frameCounter = 0;
		int sphereCount;

        GLuint renderWidth = 800;
        GLuint renderHeight = 600;
		GLuint renderSamples = 8;
		GLuint renderBounces = 4;
		GLuint renderVBO;

		GLfloat deltaTime = 0.0f;
		GLfloat lastFrame = 0.0f;

		SphereObject* spheresList;

		float3* outputBuffer;
        float3* accumulationBuffer;

        cudaGraphicsResource* cudaGRBuffer;

        cudaStream_t cudaDataStream;

		GUI lumenGUI;
};

#endif // RENDERER_H
