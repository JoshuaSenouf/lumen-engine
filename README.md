LumenEngine
======

LumenEngine is a C++ CUDA graphics engine that aimed to produce photorealistic images using Path Tracing techniques on GPU.

Screenshots
------

* Cornell :

![](https://image.ibb.co/nGCnCk/lumen_Cornell1.png)

* Depth of Field :

![](https://image.ibb.co/kvzZsk/lumen_Dof1.png)

* Low light :

![](https://image.ibb.co/fanb55/lumenlowlight1.png)


Features
------

* Rendering :
	* Progressive rendering
	* **TODO :** Render to PPM (will be added back)

* Camera :
    * Movements
	* Subpixel jitter antialiasing
	* Depth of Field (using aperture radius and focal distance)

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
