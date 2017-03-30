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
#include <tuple>

#include "src/cuda/cutil_math.h"

struct SphereObject;


class sceneParser
{
    public:
        sceneParser();
        std::tuple<SphereObject*, int> loadScene(const char* scenePath, SphereObject *spheres);
        std::string purgeString(std::string  bloatedString);
        float3 stringToFloat3(std::string vecString);
};

#endif // SCENEPARSER_H
