#ifndef LUMENGL_H
#define LUMENGL_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>


class QOpenGLShaderProgram;

class LumenGL : public QOpenGLWidget, protected QOpenGLFunctions
{
        Q_OBJECT

    public:
        LumenGL(QWidget *parentWidget = 0);
        ~LumenGL();
        void initializeGL();
        void resizeGL(int width, int height);
        void paintGL();
        void cleanGL();
};

#endif // LUMENGL_H
