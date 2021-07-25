#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include  <iostream>

#include "ui_mainwindow.h"
#include <QMainWindow>
#include <QLabel>
#include <QMessageBox>
#include <QTextEdit>
#include <QDebug>
#include <QFileDialog>
#include <QTableWidget>
#include <QTableView>
#include <QStandardItemModel>
#include <QStandardItem>


QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr):
        QMainWindow(parent)
      ,ui(new Ui::MainWindow)
  {

       ui->setupUi(this);
      connect(ui->pushButton,&QPushButton::clicked,this,&MainWindow::pushed);
      connect(ui->deleteButton,&QPushButton::clicked,this,&MainWindow::delete_row);
      connect(this,&MainWindow::send_print,this,&MainWindow::print_slot);

        ui->tableWidget = new QTableWidget(1,2);
      ui->tableView->setGridStyle(Qt::DotLine);
        ui->tableView->setShowGrid(true);
    model = new QStandardItemModel();
    labels = QObject::tr("程序名,重启次数").simplified().split(",");

  }
    ~MainWindow()
    {
        delete ui;
    }

signals:

    void send_print(QString);

public slots:
    void pushed()
    {
        std::cout<<"pushed\n";
        qDebug() << "you just clicked ok";

        QString fileName = QFileDialog::getOpenFileName(
                this,
                tr("open a file."),
                "/",
                tr("images(*.exe *jpeg *bmp);;All files(*.*)"));
            if (fileName.isEmpty()) {
                QMessageBox::warning(this, "Warning!", "Failed to open the video!");
            }

   //     emit send_print(fileName);

      // QListWidgetItem *item = new QListWidgetItem;
     //  item->setText(fileName);                    //设置列表项的文本
     //  ui->listWidget->addItem(item);          //加载列表项到列表框

     //  QTableWidgetItem *itemt = new QTableWidgetItem();
     //  itemt->setFlags(
     //              Qt::ItemIsSelectable
      //           | Qt::ItemIsEditable
      //           | Qt::ItemIsEnabled);
     //  itemt->setText("test");
      int row = ui->tableWidget->rowCount();
      // std::cout<<std::to_string(row)<<std::endl;
     //  ui->tableWidget->insertRow(row);
      // ui->tableWidget->setRowCount(2);
      // ui->tableWidget->setColumnCount(2);
        ui->tableWidget->setRowHeight(0,50);
       ui->tableWidget->setItem(0,0,new QTableWidgetItem("test"));

       ui->tableWidget->setItem(0,1,new QTableWidgetItem("test1"));


       //----------------------------------------------------



          model->setHorizontalHeaderLabels(labels);
          int rows =  model->rowCount();
         // emit send_print(QString(std::to_string(rows).c_str()));
          //定义item
          QStandardItem* item = new QStandardItem(fileName);
          model->setItem(rows,0,item);


          ui->tableView->setModel(model);

    }


    void delete_row()
    {

        int row= ui->tableView->currentIndex().row();
        emit send_print(QString(("delete row :"+std::to_string(row)).c_str()));
        model->removeRow(row);
        ui->tableView->setModel(model);
    }

    void print_slot(QString mess)
    {
        ui->plainTextEdit->appendPlainText(mess);
    }


private:
    Ui::MainWindow *ui;
    QStandardItemModel* model;
    QStringList labels;
};
#endif // MAINWINDOW_H
