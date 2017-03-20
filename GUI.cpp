#include "GUI.h"
#include "renderKernel.h"


GUI::GUI(QWidget *parent) : QWidget(parent)
{
    //------------------------------------------------------------------------------------
    //------------------------------ Widgets declaration  --------------------------------
    //------------------------------------------------------------------------------------

    windowGrid = new QGridLayout;

    renderButton = new QPushButton("RENDER TO CUDA");

    widthLabel = new QLabel("Width : ");
    heightLabel = new QLabel("Height : ");
    samplesLabel = new QLabel("Samples : ");
    bouncesLabel = new QLabel("Bounces : ");

    widthEdit = new QLineEdit("800");
    heightEdit = new QLineEdit("600");
    samplesEdit = new QLineEdit("256");
    bouncesEdit = new QLineEdit("4");

    logText = new QTextEdit();

    logFile = new QFile("renderLog.txt");

    //------------------------------------------------------------------------------------
    //----------------------------- Widgets configuration  -------------------------------
    //------------------------------------------------------------------------------------

    logText->setReadOnly(true);

    QFileInfo renderLogPath("renderLog.txt");
    if (renderLogPath.exists() && renderLogPath.isFile())
    {
        logFile->remove();
    }

    if (logFile->open(QIODevice::ReadWrite))
    {
        QTextStream logSetup(logFile);
        logSetup << "//////////////////////" << endl;
        logFile->close();
    }

    //------------------------------------------------------------------------------------
    //------------------------------ Connects declaration  -------------------------------
    //------------------------------------------------------------------------------------

    connect(renderButton, SIGNAL(clicked()), this, SLOT(callCudaRender()));

    //------------------------------------------------------------------------------------
    //-------------------------- Assign widgets to their layout --------------------------
    //------------------------------------------------------------------------------------

    windowGrid->addWidget(widthLabel, 0, 0, 1, 1);
    windowGrid->addWidget(widthEdit, 0, 1, 1, 1);
    windowGrid->addWidget(heightLabel, 1, 0, 1, 1);
    windowGrid->addWidget(heightEdit, 1, 1, 1, 1);
    windowGrid->addWidget(samplesLabel, 2, 0, 1, 1);
    windowGrid->addWidget(samplesEdit, 2, 1, 1, 1);
    windowGrid->addWidget(bouncesLabel, 3, 0, 1, 1);
    windowGrid->addWidget(bouncesEdit, 3, 1, 1, 1);
    windowGrid->addWidget(renderButton, 4, 0, 1, 2);
    windowGrid->addWidget(logText, 5, 0, 1, 2);

    this->resize(500, 350);
    this->setLayout(windowGrid);
}


GUI::~GUI()
{

}


void GUI::callCudaRender()
{
    std::freopen("renderLog.txt", "a+", stdout);

    lumenRender(widthEdit->text().toInt(), heightEdit->text().toInt(), samplesEdit->text().toInt(), bouncesEdit->text().toInt());

    std::fclose(stdout);

    logFile->open(QIODevice::ReadOnly);
    QByteArray outputFile = logFile->readAll();
    logFile->close();

    QString outputText(outputFile);
    logText->setText(outputText);
}
