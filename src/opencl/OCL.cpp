#include "OCL.h"


OCL::OCL()
{
	
}


void OCL::initOCL()
{
	setupPlatform();
	setupDevice();
	setupContext();
	setupKernel();
}


void OCL::cleanOCL()
{
	delete lumenCPUOutput;
}


void OCL::renderOCL(int renderWidth, int renderHeight, int renderSamples, int renderBounces)
{
	Scene testScene;
	testScene.loadScene("res/scenes/cornellSceneCL.txt");

	lumenSphereCount = testScene.sceneSphereCount;
	SphereObject* sceneSphereList = testScene.sceneSpheres;

	//std::cout << "COUNT : " << lumenSphereCount << std::endl;
	//for(int i = 0; i < lumenSphereCount; i++)
	//{
	//    std::cout << "RADIUS : " << sceneSphereList[i].radius << std::endl;
	//    std::cout << "POS X : " << sceneSphereList[i].position.x << " POS Y : " << sceneSphereList[i].position.y << " POS Z : " << sceneSphereList[i].position.z << std::endl;
	//    std::cout << "COL R : " << sceneSphereList[i].color.x << " COL G : " << sceneSphereList[i].color.y << " COL B: " << sceneSphereList[i].color.z << std::endl;
	//    std::cout << "EMI R : " << sceneSphereList[i].emissiveColor.x << " EMI G : " << sceneSphereList[i].emissiveColor.y << " EMI B : " << sceneSphereList[i].emissiveColor.z << std::endl;
	//    std::cout << "///////////////" << std::endl;
	//}

	// CPU-side memory allocation for the output
	lumenCPUOutput = new cl_float3[renderWidth * renderHeight];

	// Buffers init on our OpenCL device
	lumenOCLOutput = cl::Buffer(lumenOCLContext, CL_MEM_WRITE_ONLY, renderWidth * renderHeight * sizeof(cl_float3));
	lumenOCLSphereList = cl::Buffer(lumenOCLContext, CL_MEM_READ_ONLY, lumenSphereCount * sizeof(SphereObject));
	lumenOCLQueue.enqueueWriteBuffer(lumenOCLSphereList, CL_TRUE, 0, lumenSphereCount * sizeof(SphereObject), sceneSphereList);

	// We set our args for our OpenCL kernel
	lumenOCLKernel.setArg(0, lumenOCLSphereList);
	lumenOCLKernel.setArg(1, renderWidth);
	lumenOCLKernel.setArg(2, renderHeight);
	lumenOCLKernel.setArg(3, renderSamples);
	lumenOCLKernel.setArg(4, renderBounces);
	lumenOCLKernel.setArg(5, lumenSphereCount);
	lumenOCLKernel.setArg(6, lumenOCLOutput);

	// We set the number of threads/work items as the total number of pixels of our render
	std::size_t lumenGlobalWorkSize = renderWidth * renderHeight;
	std::size_t lumenLocalWorkSize = lumenOCLKernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(lumenOCLDevice);

	std::cout << "LOG : Kernel work group size: " << lumenLocalWorkSize << std::endl;

	// We make sure that the global work size of our OpenCL kernel is a multiple of local work size, else we make it so
	if (lumenGlobalWorkSize % lumenLocalWorkSize != 0)
	{
		lumenGlobalWorkSize = (lumenGlobalWorkSize / lumenLocalWorkSize + 1) * lumenLocalWorkSize;
	}

	std::cout << "LOG : Rendering..." << std::endl;

	// We run the kernel on our OpenCL device
	lumenOCLQueue.enqueueNDRangeKernel(lumenOCLKernel, NULL, lumenGlobalWorkSize, lumenLocalWorkSize);
	lumenOCLQueue.finish();

	std::cout << "LOG : Rendering done !" << std::endl;

	// We copy the result of our OpenCL kernel on our CPU
	lumenOCLQueue.enqueueReadBuffer(lumenOCLOutput, CL_TRUE, 0, renderWidth * renderHeight * sizeof(cl_float3), lumenCPUOutput);

	// We convert our OpenCL output as a readable PPM image file
	std::cout << "LOG : Saving..." << std::endl;
	clToPPM(renderWidth, renderHeight);
	std::cout << "LOG : Render successfuly saved as lumenRender.ppm !" << std::endl;

	cleanOCL();
}


void OCL::setupPlatform()
{
	// Get available platforms
	std::vector<cl::Platform> platformList;
	cl::Platform::get(&platformList);

	std::cout << "List of OpenCL platforms: " << std::endl << std::endl;

	for (int i = 0; i < platformList.size(); i++)
		std::cout << "\t" << i + 1 << ": " << platformList[i].getInfo<CL_PLATFORM_NAME>() << std::endl;

	getPlatforms(lumenOCLPlatform, platformList);

	std::cout << "\nThe following platform will be used: \t" << lumenOCLPlatform.getInfo<CL_PLATFORM_NAME>() << std::endl;
}


void OCL::getPlatforms(cl::Platform& platform, const std::vector<cl::Platform>& platforms)
{
	if (platforms.size() == 1)
	{
		platform = platforms[0];
	}

	else
	{
		int input = 0;
		std::cout << "\nPlease choose an OpenCL platform : ";
		std::cin >> input;

		while (input < 1 || input > platforms.size())
		{
			std::cin.clear();
			std::cin.ignore(std::cin.rdbuf()->in_avail(), '\n');
			std::cout << "ERROR : No such option. Please choose an actual OpenCL platform : ";
			std::cin >> input;
		}

		platform = platforms[input - 1];
	}
}


void OCL::setupDevice()
{
	// Get available OpenCL devices on our platform
	std::vector<cl::Device> devices;
	lumenOCLPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

	std::cout << "Available OpenCL devices on this platform : " << std::endl << std::endl;
	for (int i = 0; i < devices.size(); i++)
	{
		std::cout << "\t" << i + 1 << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
    }

	// Pick one device
	getDevices(lumenOCLDevice, devices);

	std::cout << "\nThe following device will be used : \t" << lumenOCLDevice.getInfo<CL_DEVICE_NAME>() << std::endl;
	std::cout << "\t\t\tLOG : Maximum compute units : " << lumenOCLDevice.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
	std::cout << "\t\t\tLOG : Maximum work group size : " << lumenOCLDevice.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
}


void OCL::getDevices(cl::Device& device, const std::vector<cl::Device>& devices)
{
	if (devices.size() == 1)
	{
		device = devices[0];
	}

	else
	{
		int input = 0;
		std::cout << "\nPlease choose an OpenCL device : ";
		std::cin >> input;

		while (input < 1 || input > devices.size()) // Simple check to be sure that the user is not choosing an option that does not exist
		{
			std::cout << "ERROR : No such option. Please choose an actual OpenCL device : ";
			std::cin.clear();
			std::cin.ignore(std::cin.rdbuf()->in_avail(), '\n');
			std::cin >> input;
		}

		device = devices[input - 1];
	}
}


void OCL::setupContext()
{
	// Create an OpenCL context and command queue on our device
	lumenOCLContext = cl::Context(lumenOCLDevice);
	lumenOCLQueue = cl::CommandQueue(lumenOCLContext, lumenOCLDevice);
}


void OCL::setupKernel()
{
	// Convert the OpenCL kernel to a readable string
	std::string source;
	std::ifstream file("src/opencl/lumenKernel.cl");

	if (!file)
	{
		std::cout << "\nNo OpenCL file found !" << std::endl << "Aborting..." << std::endl;
	}

	while (!file.eof())
	{
		char line[256];
		file.getline(line, 255);
		source += line;
	}

	lumenOCLKernelSource = source.c_str();

	 // We compile our OpenCL kernel on runtime
	lumenOCLProgram = cl::Program(lumenOCLContext, lumenOCLKernelSource);
	cl_int result = lumenOCLProgram.build({ lumenOCLDevice });

	if (result)
	{
		std::cout << "ERROR : A problem occured during the compilation of the OpenCL kernel !\n (" << result << ")" << std::endl;
	}

	if (result == CL_BUILD_PROGRAM_FAILURE)
	{
		printLog(lumenOCLProgram, lumenOCLDevice);
	}

	// We define the function that will be the entry point of our OpenCL kernel
	lumenOCLKernel = cl::Kernel(lumenOCLProgram, "lumenRender");
}


void OCL::printLog(const cl::Program& program, const cl::Device& device)
{
	std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);

	std::cerr << "OCL Log :" << std::endl << buildlog << std::endl;
}


void OCL::clToPPM(int renderWidth, int renderHeight)
{
	FILE *f = fopen("lumenRender.ppm", "w");
	fprintf(f, "P3\n%d %d\n%d\n", renderWidth, renderHeight, 255);

	for (int i = 0; i < renderWidth * renderHeight; i++)
	{
		fprintf(f, "%d %d %d ", int(lumenCPUOutput[i].s[0]), int(lumenCPUOutput[i].s[1]), int(lumenCPUOutput[i].s[2]));
	}

	fclose(f);
}
