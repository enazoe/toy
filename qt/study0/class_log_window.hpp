#ifndef LOGWINDOW_HPP
#define LOGWINDOW_HPP

#include <QPlainTextEdit>

class MyLogWindow : public QPlainTextEdit
{
    Q_OBJECT
/* snip */
public:
    void appendMessage(const QString& text)
    {
        this->appendPlainText(text); // Adds the message to the widget
    }

private:
};


#endif // LOGWINDOW_HPP
