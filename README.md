LumenEngine
======

LumenEngine is a C++ CUDA graphics engine that aimed to produce photorealistic images using Path Tracing techniques on GPU.

Screenshot
------

* Scene rendered at 720p, 8 samples, 4 bounces, in about 30 seconds using progressive rendering on a Nvidia GTX 970 :
![](https://image.ibb.co/egmKsk/lumenlowlight2.png)

Features
------

* Rendering :
	* Progressive rendering
	* **TODO :** Render to PPM (will be added back)

* Camera :
    * Movements
	* Subpixel jitter antialiasing
	* Depth of Field

* Material :
	* Perfect diffuse
	* Perfect specular (mirror)
	* Phong metal
	* Glossy/coat
	* **TODO :** Refraction/Caustics
	* **TODO :** Subsurface scattering

* Light sources :
    * Emissive spheres
    * Sky light

* Shapes :
    * Spheres
    * **TODO :** Planes/boxes
    * **TODO :** Triangles

* Acceleration structure :
	* **TODO :** BVH
 
* Utility :
    * GUI using ImGui

How to use
------

GLEngine was written using Windows/Linux, VS2015/QtCreator as the IDE, CMake as the building tool, CUDA 7.x/Compute Capabilities 5.2 (can be probably lowered to 3.2 if your hardware is not recent enough) as the GPGPU API, and a C++11 compiler in mind.

Download the source, build the project structure using CMake 3.x, open the project using your favorite IDE (tested on VS2015/QtCreator), build the project, and everything should be ready to use.

Dependencies
------

- Window & Input system : GLFW
- GUI system : dear imgui
- OpenGL Function Loader : GLAD
- OpenGL Mathematic Functions : GLM
- GPGPU : CUDA 7.x

Credits
------

- Samuel Lapere
- Peter Kutz/Yining Karl Li
- RichieSams
