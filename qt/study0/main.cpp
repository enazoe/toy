#include "mainwindow.hpp"

#include <QApplication>
#include <QLocale>
#include <QTranslator>
#include <QStringList>
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);


    MainWindow w;
    w.show();
    return a.exec();
}
