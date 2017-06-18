#include "renderer.h"


Renderer::Renderer()
{
	
}


int Renderer::runRenderer()
{
    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
//    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    GLFWwindow* window = glfwCreateWindow(renderWidth, renderHeight, "LumenEngine", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSwapInterval(1);

    gladLoadGL();

    glViewport(0, 0, renderWidth, renderHeight);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glMatrixMode(GL_PROJECTION);
    glOrtho(0.0, renderWidth, 0.0, renderHeight, 1.0, -1.0);

	initCUDAData();

    ImGui_ImplGlfwGL3_Init(window, true);
	lumenGUI.setRenderResolution(renderWidth, renderHeight);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);


    while (!glfwWindowShouldClose(window))
    {
        GLfloat currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        glfwPollEvents();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //--------------
        // GUI setting
        //--------------
        lumenGUI.setupGUI();

        frameCounter++;

        cudaStreamCreate(&cudaDataStream); // Ensure data synchronization with the CUDA device
        cudaGraphicsMapResources(1, &cudaGRBuffer, cudaDataStream); // Map the graphics resource (indirectly our VBO) to CUDA so that the kernel can use it

        lumenRender(outputBuffer, accumulationBuffer, renderWidth, renderHeight, renderSamples, renderBounces, sphereCount, spheresList, frameCounter); // Send the needed the CPU data to the CUDA kernel

        cudaGraphicsUnmapResources(1, &cudaGRBuffer, 0); // Unmap the graphics resource so that OpenGL can use them
        cudaStreamDestroy(cudaDataStream);

		displayGLBuffer(); // Display the data inside the VBO

        //----------------
        // GUI rendering
        //----------------
        lumenGUI.renderGUI();

        glfwSwapBuffers(window);
    }

    //---------
    // Cleaning
    //---------
    lumenGUI.stopGUI();
	cleanCUDAData();
	cleanCUDAScene();

    glfwTerminate();

	return 0;
}


void Renderer::initCUDAData()
{
	initCUDAScene(); // Set up the selected scene and allocate the needed data for CUDA in memory

	cudaMalloc(&accumulationBuffer, renderWidth * renderHeight * sizeof(float3)); // Set memory for the Accumulation Buffer

	initRenderVBO(&renderVBO, &cudaGRBuffer, cudaGraphicsRegisterFlagsNone); // Create the OpenGL VBO that will be used to store the result from CUDA so that we can display it

	cudaStreamCreate(&cudaDataStream); // Set the synchronization stream
	cudaGraphicsMapResources(1, &cudaGRBuffer, cudaDataStream); // Map the graphics resource to CUDA (indirectly our VBO)

	size_t bytes;
	cudaGraphicsResourceGetMappedPointer((void**)&outputBuffer, &bytes, cudaGRBuffer); // Set the access (read/write) to the CUDA graphics resource using our outputBuffer

	cudaGraphicsUnmapResources(1, &cudaGRBuffer, cudaDataStream); // Unmap the graphics resource from CUDA
	cudaStreamDestroy(cudaDataStream); // Free the synchronization stream
}


void Renderer::cleanCUDAData()
{
	cleanRenderVBO(&renderVBO, cudaGRBuffer);
	cudaFree(accumulationBuffer);
}


void Renderer::initRenderVBO(GLuint* renderVBO, cudaGraphicsResource **cudaGRBuffer, unsigned int BufferFlags)
{
    glGenBuffers(1, renderVBO);
    glBindBuffer(GL_ARRAY_BUFFER, *renderVBO);

    unsigned int vboSize = renderWidth * renderHeight * sizeof(float3);
    glBufferData(GL_ARRAY_BUFFER, vboSize, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(cudaGRBuffer, *renderVBO, BufferFlags);
}


void Renderer::cleanRenderVBO(GLuint* renderVBO, cudaGraphicsResource* cudaGRBuffer)
{
	cudaGraphicsUnregisterResource(cudaGRBuffer);

	glBindBuffer(GL_ARRAY_BUFFER, *renderVBO);
	glDeleteBuffers(1, renderVBO);

	*renderVBO = 0;
}


void Renderer::initCUDAScene()
{
    Scene testScene;
    testScene.loadScene("res/scenes/cornellSceneCUDA.txt");

    sphereCount = testScene.getSphereCount();
    SphereObject* sceneSpheres = testScene.getSceneSpheresList();

	//std::cout << "SPHERECOUNT : " << sceneSphere[i].radius << std::endl;
 //   for(int i = 0; i < sphereCount; i++)
 //   {
 //       std::cout << "RADIUS : " << sceneSphere[i].radius << std::endl;
 //       std::cout << "POS X : " << sceneSphere[i].position.x << " POS Y : " << sceneSphere[i].position.y << " POS Z : " << sceneSphere[i].position.z << std::endl;
 //       std::cout << "COL R : " << sceneSphere[i].color.x << " COL G : " << sceneSphere[i].color.y << " COL B: " << sceneSphere[i].color.z << std::endl;
 //       std::cout << "EMI R : " << sceneSphere[i].emissiveColor.x << " EMI G : " << sceneSphere[i].emissiveColor.y << " EMI B : " << sceneSphere[i].emissiveColor.z << std::endl;
 //       std::cout << "///////////////" << std::endl;
 //   }

    cudaMalloc(&spheresList, (sphereCount) * sizeof(SphereObject));
    cudaMemcpy(spheresList, sceneSpheres, (sphereCount) * sizeof(SphereObject), cudaMemcpyHostToDevice);

	delete[] sceneSpheres;
	sceneSpheres = NULL;
}


void Renderer::cleanCUDAScene()
{
	sphereCount = 0;
	cudaFree(spheresList);
}


void Renderer::displayGLBuffer() // Currently using the old OpenGL pipeline, should switch to using actual VAO/VBOs and a shader in order to render to a framebuffer and display the result
{
	glBindBuffer(GL_ARRAY_BUFFER, renderVBO);
	glVertexPointer(2, GL_FLOAT, 12, 0);
	glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, renderWidth * renderHeight);
	glDisableClientState(GL_VERTEX_ARRAY);
}