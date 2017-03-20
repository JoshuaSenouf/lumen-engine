#include <QApplication>
#include "renderKernel.h"
#include "GUI.h"


int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    app.setStyle("Fusion");

    GUI window;
    window.show();

    return app.exec();
}
