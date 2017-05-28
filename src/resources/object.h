#ifndef OBJECT_H
#define OBJECT_H

#include "material.h"
#include <CL\cl.hpp>


struct RayObject
{
        cl_float3 origin;
		cl_float3 direction;
};


struct SphereObject	// Using dummys to not corrupt the memory alignment in the OpenCL kernel is disgusting as hell, need to find an alternative
{
		cl_float radius;
		cl_float dummy1;
		cl_float dummy2;
		cl_float dummy3;
		cl_float3 position;
		cl_float3 color;
		cl_float3 emissiveColor;
		enum materialType material;
		cl_float dummy4;
		cl_float dummy5;
		cl_float dummy6;
};


#endif // OBJECT_H
