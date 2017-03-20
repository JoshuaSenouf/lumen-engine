#ifndef GUI_H
#define GUI_H

#include <QtWidgets>


class GUI : public QWidget
{
        Q_OBJECT

    public:
        GUI(QWidget *parent = 0);
        ~GUI();

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
};


#endif // GUI_H
