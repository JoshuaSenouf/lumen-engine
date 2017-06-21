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

    window = glfwCreateWindow(renderWidth, renderHeight, "LumenEngine", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSwapInterval(1);

    gladLoadGL();

    glViewport(0, 0, renderWidth, renderHeight);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glMatrixMode(GL_PROJECTION);
    glOrtho(0.0, renderWidth, 0.0, renderHeight, 1.0, -1.0);


    cudaCamera.setCamera(glm::vec2(renderWidth, renderHeight));

    initCUDAScene();
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
        // GUI setting & callbacks
        //--------------
        lumenGUI.setupGUI();

        ImGuiIO& io = ImGui::GetIO();
        ImVec2 currentMousePos = ImGui::GetMousePos();

        keyboardCallback(&io); // Currently checking this at every frame, need to find a way to see if an actual key has been pressed, just like GLFW KeyCallback

        if (lastPosX !=  currentMousePos.x || lastPosY != currentMousePos.y)
            mouseCallback(&io, currentMousePos.x, currentMousePos.y);

        //--------------
        // CUDA Rendering
        //--------------
        if (renderReset) // If anything in camera or scene data has changed, we flush the CUDA data and reinit them again
            resetRender();

        frameCounter++;

        cudaStreamCreate(&cudaDataStream); // Ensure data synchronization with the CUDA device
        cudaGraphicsMapResources(1, &cudaGRBuffer, cudaDataStream); // Map the graphics resource (indirectly our VBO) to CUDA so that the kernel can use it

        lumenRender(outputBuffer, accumulationBuffer, renderWidth, renderHeight, renderSamples, renderBounces, sphereCount, spheresList, frameCounter, cudaCameraInfo); // Send the needed the CPU data to the CUDA kernel

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
    cudaMalloc(&accumulationBuffer, renderWidth * renderHeight * sizeof(glm::vec3)); // Set memory for the Accumulation Buffer

    cudaMalloc(&cudaCameraInfo, sizeof(CameraInfo));
    cudaMemcpy(cudaCameraInfo, cudaCamera.getCameraInfo(), sizeof(CameraInfo), cudaMemcpyHostToDevice);

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
    cudaFree(cudaCameraInfo);
}


void Renderer::initRenderVBO(GLuint* renderVBO, cudaGraphicsResource **cudaGRBuffer, unsigned int BufferFlags)
{
    glGenBuffers(1, renderVBO);
    glBindBuffer(GL_ARRAY_BUFFER, *renderVBO);

    unsigned int vboSize = renderWidth * renderHeight * sizeof(glm::vec3);
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
    testScene.loadScene("res/scenes/testScene.txt");

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


void Renderer::resetRender()
{
    cleanCUDAData();
    frameCounter = 0;
    initCUDAData();

    renderReset = false;
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


void Renderer::keyboardCallback(ImGuiIO* guiIO)
{
    if (guiIO->KeysDown[GLFW_KEY_ESCAPE])
        glfwSetWindowShouldClose(window, GL_TRUE);

    if (guiIO->KeysDown[GLFW_KEY_W])
    {
        cudaCamera.keyboardCall(FORWARD, deltaTime);
        renderReset = true;
    }

    if (guiIO->KeysDown[GLFW_KEY_S])
    {
        cudaCamera.keyboardCall(BACKWARD, deltaTime);
        renderReset = true;
    }

    if (guiIO->KeysDown[GLFW_KEY_A])
    {
        cudaCamera.keyboardCall(LEFT, deltaTime);
        renderReset = true;
    }

    if (guiIO->KeysDown[GLFW_KEY_D])
    {
        cudaCamera.keyboardCall(RIGHT, deltaTime);
        renderReset = true;
    }

    if (guiIO->KeysDown[GLFW_KEY_KP_ADD])
    {
        if (guiIO->KeysDown[GLFW_KEY_LEFT_CONTROL])
            cudaCamera.setCameraFocalDistance(cudaCamera.getCameraFocalDistance() + 0.1f);
        else
            cudaCamera.setCameraApertureRadius(cudaCamera.getCameraApertureRadius() + 0.005f);

        renderReset = true;
    }

    if (guiIO->KeysDown[GLFW_KEY_KP_SUBTRACT])
    {
        if (guiIO->KeysDown[GLFW_KEY_LEFT_CONTROL])
            cudaCamera.setCameraFocalDistance(cudaCamera.getCameraFocalDistance() - 0.1f);
        else
            cudaCamera.setCameraApertureRadius(cudaCamera.getCameraApertureRadius() - 0.005f);

        renderReset = true;
    }
}


void Renderer::mouseCallback(ImGuiIO* guiIO, float mousePosX, float mousePosY)
{
    if (firstMouse)
    {
        lastPosX = mousePosX;
        lastPosY = mousePosY;
        firstMouse = false;
    }

    float offsetX = mousePosX - lastPosX;
    float offsetY = mousePosY - lastPosY;

    lastPosX = mousePosX;
    lastPosY = mousePosY;

    if (guiIO->MouseDown[GLFW_MOUSE_BUTTON_RIGHT])
    {
        if (offsetX != 0 || offsetY != 0)
        {
            cudaCamera.mouseCall(offsetX, offsetY, true);
            renderReset = true;
        }
    }
}