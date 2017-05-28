#ifndef SCENEPARSER_H
#define SCENEPARSER_H

#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <cctype>
#include <sstream>
#include <cstring>
#include <algorithm>

#include "object.h"
#include <CL\cl.hpp>

#define float3(x, y, z) {{x, y, z}}	// Useful macro before I switch to GLM


class Scene
{
    public:
        Scene();

        void loadScene(const char* scenePath);
        std::string purgeString(std::string bloatedString);
		cl_float3 stringToFloat3(std::string vecString);

        SphereObject* sceneSpheres;
        int sceneSphereCount = 0;
};

#endif // SCENEPARSER_H
