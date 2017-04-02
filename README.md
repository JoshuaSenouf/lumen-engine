LumenEngine
======
LumenEngine is a C++ CUDA graphics engine that aimed to produce photorealistic images using Path Tracing techniques on GPU.

Screenshot
------

* Cornell Box - 4K, 1024 samples, rendered in 60 seconds using a Nvidia GTX 970 :
![](https://image.ibb.co/gHSrwF/lumen_Render.png)


How to use
------
GLEngine was written using Linux, QtCreator as the IDE, QMake as the building tool, CUDA 7.x/Compute Capabilities 5.2 (can be probably lowered to 3.2 if your hardware is not recent enough) as the GPGPU API and a C++11 compiler in mind.

Download the source, open the LumenEngine.pro file with QtCreator, build the project, and everything should be ready to use.

Dependencies
------
- Window, Input & GUI system : Qt5.x
- GPGPU : CUDA 7.x
