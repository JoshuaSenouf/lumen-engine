#ifndef SCENEPARSER_H
#define SCENEPARSER_H

#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <cctype>
#include <sstream>


class sceneParser
{
    public:
        sceneParser();
        void loadScene(const char* scenePath);
};

#endif // SCENEPARSER_H
