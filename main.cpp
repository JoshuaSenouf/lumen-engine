#include <QApplication>
#include "src/interface/GUI.h"
#include "src/scene/sceneParser.h"


int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    app.setStyle("Fusion");

    sceneParser testScene;
    testScene.loadScene("res/scenes/testScene.txt");

    GUI window;
    window.show();

    return app.exec();
}
