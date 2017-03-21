#include "src/scene/sceneParser.h"


sceneParser::sceneParser()
{

}

void sceneParser::loadScene(const char* scenePath)
{
    std::ifstream sceneReader(scenePath);
    std::string currentLine;
    std::string objName, objType, radius, position, color, emissiveColor, materialType;

    while(getline(sceneReader, currentLine))
    {
        if(!currentLine.empty())
        {
            std::stringstream iss(currentLine);

            getline(iss, objType, ':');
            getline(iss, objName, ';');

            if(objType == "SPHERE")
            {
                getline(iss, radius, ';');
            }

            getline(iss, position, ';');
            getline(iss, color, ';');
            getline(iss, emissiveColor, ';');
            getline(iss, materialType, ';');

            std::cout << "OBJECT TYPE : " << objType << std::endl;
            std::cout << "OBJECT NAME : " << objName << std::endl;
            if(objType == "SPHERE")
                std::cout << "RADIUS : " << radius << std::endl;
            std::cout << "POSITION : " << position << std::endl;
            std::cout << "COLOR : " << color << std::endl;
            std::cout << "EMISSIVE COLOR : " << emissiveColor << std::endl;
            std::cout << "MATERIAL TYPE : " << materialType << std::endl;
            std::cout << "=================" << std::endl;
        }
    }

    sceneReader.close();
}
