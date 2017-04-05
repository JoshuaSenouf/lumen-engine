#include "src/interface/lumenGUI.h"


LumenGUI::LumenGUI(QWidget *parent) : QWidget(parent)
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

    cudaRender = new LumenCUDA();

    glFormat = new QSurfaceFormat;

    glRender = new LumenGL(this);

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
        logSetup << "//////////////////////\n" << endl;
        logFile->close();
    }

    glFormat->setRenderableType(QSurfaceFormat::OpenGL);
    glFormat->setProfile(QSurfaceFormat::CoreProfile);
    glFormat->setVersion(4, 0);

    glRender->setFormat(*glFormat);
    glRender->resize(QSize(800, 600));
    glRender->setWindowTitle("LumenEngine - CUDA/GL");
    glRender->show();

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

    this->setLayout(windowGrid);
    this->setWindowTitle("LumenEngine - GUI");
    this->resize(640, 480);
}


LumenGUI::~LumenGUI()
{

}


void LumenGUI::callCudaRender()
{
    glRender->resize(QSize(widthEdit->text().toInt(), heightEdit->text().toInt()));

    std::freopen("renderLog.txt", "a+", stdout);

    cudaRender->render(widthEdit->text().toInt(), heightEdit->text().toInt(), samplesEdit->text().toInt(), bouncesEdit->text().toInt());

    std::fclose(stdout);

    logFile->open(QIODevice::ReadOnly);
    QByteArray outputFile = logFile->readAll();
    logFile->close();

    QString outputText(outputFile);
    logText->setText(outputText);
}
