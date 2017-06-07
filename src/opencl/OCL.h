#ifndef OCL_H
#define OCL_H

#include <iostream>
#include <fstream>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_WARNINGS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <CL\cl.hpp>
#include "scene.h"
#include "object.h"
#include <typeinfo>


class OCL
{
    public:
        OCL();

		void initOCL();
		void cleanOCL();
		void renderOCL(int renderWidth, int renderHeight, int renderSamples, int renderBounces);
		void setupPlatform();
		void getPlatforms(cl::Platform& platform, const std::vector<cl::Platform>& listPlatform);
		void setupDevice();
		void getDevices(cl::Device& device, const std::vector<cl::Device>& listDevice);
		void setupContext();
		void setupKernel();
		void printLog(const cl::Program& program, const cl::Device& device);
		void clToPPM(int renderWidth, int renderHeight);

		const char* lumenOCLKernelSource;

		cl_float4* lumenCPUOutput;

		cl::Platform lumenOCLPlatform;
		cl::Device lumenOCLDevice;
		cl::Context lumenOCLContext;
		cl::CommandQueue lumenOCLQueue;
		cl::Program lumenOCLProgram;
		cl::Kernel lumenOCLKernel;
		cl::Buffer lumenOCLOutput;
		cl::Buffer lumenOCLSphereList;

		SphereObject* spheres;
		int lumenSphereCount;
};

#endif // OCL_H
