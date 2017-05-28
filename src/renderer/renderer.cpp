#include "renderer.h"


Renderer::Renderer()
{
	
}


void Renderer::initRenderer()
{

}


int Renderer::runRenderer()
{
	lumenOCL.initOCL();

	lumenGUI.setRenderResolution(renderWidth, renderHeight);
	lumenGUI.setOCL(&lumenOCL);

	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	GLFWwindow* window = glfwCreateWindow(renderWidth, renderHeight, "LumenEngine", nullptr, nullptr);
	glfwMakeContextCurrent(window);

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSwapInterval(1);

	gladLoadGL();

	glViewport(0, 0, renderWidth, renderHeight);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	ImGui_ImplGlfwGL3_Init(window, true);

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
	glfwTerminate();

	return 0;
}