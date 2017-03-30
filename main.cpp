#include <QApplication>
#include "src/interface/GUI.h"


int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    app.setStyle("Fusion");

    GUI window;
    window.show();

    return app.exec();
}
