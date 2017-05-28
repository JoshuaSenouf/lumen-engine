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

#include "gui.h"
#include "OCL.h"


class Renderer
{
    public:
        Renderer();

		void initRenderer();
		int runRenderer();

		GLfloat deltaTime = 0.0f;
		GLfloat lastFrame = 0.0f;

		bool firstMouse = true;
		bool guiIsOpen = true;
		bool keys[1024];

		GLuint renderWidth = 800;
		GLuint renderHeight = 600;

		GUI lumenGUI;
		OCL lumenOCL;
};

#endif // RENDERER_H
