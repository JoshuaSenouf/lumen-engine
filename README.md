LumenEngine
======

LumenEngine is a C++ CUDA/OpenCL graphics engine that aimed to produce photorealistic images using Path Tracing techniques on GPU.

Screenshot
------

* Cornell Box - 720p, 16 samples, rendered using progressive rendering on a Nvidia GTX 970 :
![](https://image.ibb.co/kk6NtQ/lumen_Engine1.png)

Features
------

* Rendering :
	* Progressive rendering
	* **TODO :** Render to PPM (will be added back)

* **TODO :**Camera :
    * **TODO :** Movements
	* **TODO :** Subpixel jitter antialiasing
	* **TODO :** Depth of Field

* Material :
	* Perfect diffuse
	* Perfect specular (mirror)
	* Phong metal
	* Glossy/coat
	* **TODO :** Refraction/Caustics
	* **TODO :** Subsurface scattering

* Shapes :
    * Spheres
    * **TODO :** Planes/boxes
    * **TODO :** Triangles

* Acceleration structure :
	* **TODO :** BVH

* Lights :
    * Emissive spheres
    * **TODO :** Sky light
 
* Utility :
    * GUI using ImGui

How to use
------

GLEngine was written using Windows/Linux, VS2015/QtCreator as the IDE, CMake as the building tool, CUDA 7.x/Compute Capabilities 5.2 (can be probably lowered to 3.2 if your hardware is not recent enough) and OpenCL 1.x as the GPGPU APIs, and a C++11 compiler in mind.

Download the source, build the project structure using CMake 3.x, open the project using your favorite IDE (tested on VS2015/QtCreator), build the project, and everything should be ready to use.

Dependencies
------

- Window & Input system : GLFW
- GUI system : dear imgui
- OpenGL Function Loader : GLAD
- OpenGL Mathematic Functions : GLM
- GPGPU : CUDA 7.x/OpenCL 1.x

Credits
------

- Samuel Lapere
- Peter Kurtz
- RichieSams
