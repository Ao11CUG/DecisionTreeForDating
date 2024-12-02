#include "DecisionTreeForDating.h"
#include <QFileDialog>
#include <QTextStream>
#include <QMessageBox>
#include <QWidget> 
#include <QPainter>
#include <QVBoxLayout>

DecisionTreeForDating::DecisionTreeForDating(QWidget *parent)
    : QMainWindow(parent), root(nullptr)
{
    ui.setupUi(this);

    // 创建和配置 QTextEdit
    ui.textEdit = new QTextEdit(this);
    ui.textEdit->setReadOnly(true);  // 设置为只读
    ui.textEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding); // 自适应大小
    ui.textEdit->setLineWrapMode(QTextEdit::WidgetWidth);  // 自动换行

    // 将 QTextEdit 添加到 drawingArea
    QVBoxLayout* layout = new QVBoxLayout(ui.drawingArea);
    layout->addWidget(ui.textEdit);
    ui.drawingArea->setLayout(layout);

    // 设置默认选择为 ID3
    ui.comboBox->setCurrentIndex(0);  // 0表示 ID3

    // 连接 QComboBox 的信号槽
    connect(ui.comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onComboBoxChanged(int)));

    connect(ui.chooseSampleButton, &QPushButton::clicked, this, &DecisionTreeForDating::chooseSample);
    connect(ui.chooseInputButton, &QPushButton::clicked, this, &DecisionTreeForDating::chooseInput);
    connect(ui.beginPredictButton, &QPushButton::clicked, this, &DecisionTreeForDating::beginPredict);

}

DecisionTreeForDating::~DecisionTreeForDating()
{}

// 处理 QComboBox 的选择改变
void DecisionTreeForDating::onComboBoxChanged(int index) {
    // 根据 QComboBox 的选项设置 useC45
    if (index == 0) {
        // 如果选择了 ID3
        decisionTree.useC45 = false;  // 使用 ID3
    }
    else if (index == 1) {
        // 如果选择了 C4.5
        decisionTree.useC45 = true;   // 使用 C4.5
    }
}

void DecisionTreeForDating::chooseSample() {
    // 打开文件对话框选择CSV文件
    QString fileName = QFileDialog::getOpenFileName(this, QStringLiteral("选择样本数据"), "", "CSV Files (*.csv)");
    if (fileName.isEmpty()) {
        return;
    }

    QFileInfo fileInfo(fileName);
    ui.lineEditSample->setText(fileInfo.fileName());

    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::critical(this, QStringLiteral("错误"), QStringLiteral("无法打开文件！"));
        return;
    }

    QTextStream in(&file);
    vector<DataPoint> sampleData;

    // 读取CSV文件
    while (!in.atEnd()) {
        QString line = in.readLine();
        QStringList columns = line.split(",");

        if (columns.size() != 5) {
            continue;  // 假设每行有5个字段：weather, temperature, humidity, wind, label
        }

        DataPoint point;
        point.weather = columns[0].toStdString();
        point.temperature = columns[1].toStdString();
        point.humidity = columns[2].toStdString();
        point.wind = columns[3].toStdString();
        point.date = columns[4].toStdString();  // 标签

        sampleData.push_back(point);
    }

    // 保存样本数据用于训练
    decisionTree.trainData = sampleData;

    // 检查是否导入了样本数据和待预测数据
    if (decisionTree.trainData.empty()) {
        QMessageBox::warning(this, QStringLiteral("错误"), QStringLiteral("样本数据为空！"));
        return;
    }
}

void DecisionTreeForDating::chooseInput() {
    // 打开文件对话框选择待预测数据CSV文件
    QString fileName = QFileDialog::getOpenFileName(this, QStringLiteral("选择待预测数据"), "", "CSV Files (*.csv)");
    if (fileName.isEmpty()) {
        return;
    }

    QFileInfo fileInfo(fileName);
    ui.lineEditInput->setText(fileInfo.fileName());

    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::critical(this, QStringLiteral("错误"), QStringLiteral("无法打开文件！"));
        return;
    }

    QTextStream in(&file);
    vector<DataPoint> testData;

    // 读取CSV文件
    while (!in.atEnd()) {
        QString line = in.readLine();
        QStringList columns = line.split(",");

        if (columns.size() != 4) {
            continue;  // 假设每行有4个字段：weather, temperature, humidity, wind
        }

        DataPoint point;
        point.weather = columns[0].toStdString();
        point.temperature = columns[1].toStdString();
        point.humidity = columns[2].toStdString();
        point.wind = columns[3].toStdString();

        testData.push_back(point);
    }

    // 保存待预测数据
    decisionTree.testData = testData;

    // 检查是否导入了样本数据和待预测数据
    if (decisionTree.testData.empty()) {
        QMessageBox::warning(this, "Error", "Test data is empty.");
        return;
    }
}

void DecisionTreeForDating::beginPredict() {
    // 获取选中的算法类型
    QString selectedAlgorithm = ui.comboBox->currentText();

    ui.drawingArea->update();

    // 检查是否导入了样本数据和待预测数据
    if (decisionTree.trainData.empty() || decisionTree.testData.empty()) {
        QMessageBox::warning(this, "错误", QStringLiteral("样本数据或待预测数据为空！"));
        return;
    }

    // 创建决策树并训练
    vector<string> features = { "weather", "temperature", "humidity", "wind" };

    // 判断是否使用信息增益率
    if (selectedAlgorithm == "C4.5") decisionTree.useC45 == true;

    root = decisionTree.buildTree(decisionTree.trainData, features);  // 更新全局根节点

    if (!root) {
        QMessageBox::warning(this, QStringLiteral("错误"), QStringLiteral("决策树未能成功构建！"));
        return;
    }

    // 生成决策树文本并更新 QTextEdit
    QString treeText = generateTreeText(root, 0, true);
    ui.textEdit->setPlainText(treeText);  // 将文本设置到 QTextEdit

    // 创建一个字符串来保存所有的预测结果
    QString results;

    // 根据选择的算法进行预测
    for (const auto& test : decisionTree.testData) {
        string result;
        if (selectedAlgorithm == "ID3") {
            result = decisionTree.predictID3(root, test);
        }
        else if (selectedAlgorithm == "C4.5") {
            result = decisionTree.predictC45(root, test);
        }

        // 将预测结果添加到字符串中
        results += QString::fromStdString(result) + "\n";
    }

    // 输出预测结果到消息框
    QMessageBox::information(this, QStringLiteral("预测结果："), results);
}

QString DecisionTreeForDating::generateTreeText(TreeNode* node, int level, bool isLast) {
    if (!node) return "";

    QString text;
    QString indent(level * 4, ' '); // 控制文本缩进
    QString treeSymbol = isLast ? "    " : "|   "; // 控制当前层次的连接符号

    // 添加当前节点信息
    text += indent + (level > 0 ? treeSymbol : "") + QString::fromStdString(node->label.empty() ? node->feature : node->label) + "\n";

    // 遍历子节点
    for (auto it = node->children.begin(); it != node->children.end(); ++it) {
        bool lastChild = (std::next(it) == node->children.end()); // 判断是否为最后一个子节点
        text += generateTreeText(it->second, level + 1, lastChild); // 递归
    }
    return text;
}


