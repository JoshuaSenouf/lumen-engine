#include "lumenCUDA.h"


LumenCUDA::LumenCUDA()
{

}


void LumenCUDA::setScene()
{
    Scene testScene;
    testScene.loadScene("res/scenes/cornellSceneCUDA.txt");

    sphereCount = testScene.sceneSphereCount;
    SphereObject* sceneSphere = testScene.sceneSpheres;

//    for(int i = 0; i < sphereCount; i++)
//    {
//        std::cout << "RADIUS : " << sceneSphere[i].radius << std::endl;
//        std::cout << "POS X : " << sceneSphere[i].position.x << " POS Y : " << sceneSphere[i].position.y << " POS Z : " << sceneSphere[i].position.z << std::endl;
//        std::cout << "COL R : " << sceneSphere[i].color.x << " COL G : " << sceneSphere[i].color.y << " COL B: " << sceneSphere[i].color.z << std::endl;
//        std::cout << "EMI R : " << sceneSphere[i].emissiveColor.x << " EMI G : " << sceneSphere[i].emissiveColor.y << " EMI B : " << sceneSphere[i].emissiveColor.z << std::endl;
//        std::cout << "///////////////" << std::endl;
//    }

    cudaMalloc(&spheres, (sphereCount) * sizeof(SphereObject));
    cudaMemcpy(spheres, sceneSphere, (sphereCount) * sizeof(SphereObject), cudaMemcpyHostToDevice);
}


void LumenCUDA::render(int width = 800, int height = 600, int samples = 256, int bounces = 4)
{
    setScene();

    lumenRender(width, height, samples, bounces, sphereCount, spheres);
}
