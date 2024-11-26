#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_DecisionTreeForDating.h"
#include "DecisionTree.h"

class DecisionTreeForDating : public QMainWindow
{
    Q_OBJECT

public:
    DecisionTreeForDating(QWidget *parent = nullptr);
    ~DecisionTreeForDating();

public slots:
    void chooseSample();
    void chooseInput();
    void beginPredict();

private:
    Ui::DecisionTreeForDatingClass ui;

    DecisionTree decisionTree;

    void onComboBoxChanged(int index);

    TreeNode* root;  // 保存构建好的决策树的根节点

    QString generateTreeText(TreeNode* node, int level, bool isLast);
};
