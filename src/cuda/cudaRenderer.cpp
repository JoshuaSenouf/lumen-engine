#include "cudaRenderer.h"


cudaRenderer::cudaRenderer()
{

}


void cudaRenderer::setScene()
{
    spheres = (SphereObject*)malloc(sizeof(SphereObject) * 1);

    sceneParser testScene;
    std::tuple<SphereObject*, int> sceneParsing;
    sceneParsing = testScene.loadScene("res/scenes/testScene.txt", spheres);

    spheres = std::get<0>(sceneParsing);
    sphereCount = std::get<1>(sceneParsing);

//    for(int i = 0; i < sphereCount; i++)
//    {
//        std::cout << "SPHERE RADIUS : " << tempSpheres[i].radius << std::endl;
//    }
}


void cudaRenderer::render(int width = 800, int height = 600, int samples = 128, int bounces = 4)
{
    setScene();

    lumenRender(width, height, samples, bounces, sphereCount, spheres);
}
