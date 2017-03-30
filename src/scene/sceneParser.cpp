#include "src/scene/sceneParser.h"
#include "src/cuda/cudaRenderer.h"


sceneParser::sceneParser()
{

}


std::tuple<SphereObject*, int> sceneParser::loadScene(const char* scenePath, SphereObject *spheres)
{
    SphereObject tempSphere;

    int sphereCount = 0;

    std::ifstream sceneReader(scenePath);
    std::string currentLine, objType, tempString;

    while(getline(sceneReader, currentLine))
    {
        if(!currentLine.empty() && !(currentLine[0] == '#'))
        {
            std::stringstream iss(currentLine);

            getline(iss, objType, ';');

            getline(iss, tempString, ';');
            tempSphere.radius = std::stof(tempString);

            getline(iss, tempString, ';');
            tempSphere.position = stringToFloat3(purgeString(tempString));

            getline(iss, tempString, ';');
            tempSphere.color = stringToFloat3(purgeString(tempString));

            getline(iss, tempString, ';');
            tempSphere.emissiveColor = stringToFloat3(purgeString(tempString));

            getline(iss, tempString, ';');
            tempSphere.material = static_cast<materialType>(std::stoi(tempString));

            spheres[sphereCount] = tempSphere;
            sphereCount++;
            spheres = (SphereObject*)realloc(spheres, sizeof(SphereObject) * (sphereCount + 1));
        }
    }

    sceneReader.close();

    return std::tuple<SphereObject*, int>(spheres, sphereCount);
}


std::string sceneParser::purgeString(std::string bloatedString)
{
    char badChars[] = "()";

    for (unsigned int i = 0; i < strlen(badChars); ++i)
    {
       bloatedString.erase(std::remove(bloatedString.begin(), bloatedString.end(), badChars[i]), bloatedString.end());
    }

    return bloatedString;
}


float3 sceneParser::stringToFloat3(std::string vecString)
{
    int componentCount = 0;

    char vecComponents[3];

    std::ifstream vecReader(vecString);
    std::string currentValue;
    float3 cleanVec;

    while(getline(vecReader, currentValue, ','))
    {
        vecComponents[componentCount] = std::stof(currentValue);
        componentCount++;
    }

    cleanVec.x = vecComponents[0];
    cleanVec.y = vecComponents[1];
    cleanVec.z = vecComponents[2];

    return cleanVec;
}
