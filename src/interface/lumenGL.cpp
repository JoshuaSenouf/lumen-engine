#include "lumenGL.h"

#include <QDebug>
#include <QString>
#include <QOpenGLShaderProgram>


LumenGL::LumenGL(QWidget *parentWidget)
{
//    this->resize(parentWidget->size());
}


LumenGL::~LumenGL()
{
    makeCurrent();
    cleanGL();
}


void LumenGL::initializeGL()
{
    initializeOpenGLFunctions();

    glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
}


void LumenGL::resizeGL(int width, int height)
{
    glViewport(0, 0, width, height);
}


void LumenGL::paintGL()
{
    glViewport(0, 0, width(), height());
    glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}


void LumenGL::cleanGL()
{
    // PLACEHOLDER
}
