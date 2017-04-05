#ifndef LUMENGUI_H
#define LUMENGUI_H

#include <QtWidgets>
#include "lumenGL.h"
#include "src/cuda/renderKernel.h"
#include "src/cuda/lumenCUDA.h"


class LumenGUI : public QWidget
{
        Q_OBJECT

    public:
        LumenGUI(QWidget *parent = 0);
        ~LumenGUI();

    public slots:
        void callCudaRender();

    private:
        QGridLayout *windowGrid;

        QPushButton *renderButton;

        QLabel *widthLabel;
        QLabel *heightLabel;
        QLabel *samplesLabel;
        QLabel *bouncesLabel;

        QLineEdit *widthEdit;
        QLineEdit *heightEdit;
        QLineEdit *samplesEdit;
        QLineEdit *bouncesEdit;

        QTextEdit *logText;

        QFile *logFile;

        LumenCUDA *cudaRender;

        LumenGL *glRender;

        QSurfaceFormat *glFormat;
};


#endif // LUMENGUI_H
