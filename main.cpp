#include <QApplication>
#include "src/interface/lumenGUI.h"


int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    app.setStyle("Fusion");

    LumenGUI window;
    window.show();

    return app.exec();
}
