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

    // ���������� QTextEdit
    ui.textEdit = new QTextEdit(this);
    ui.textEdit->setReadOnly(true);  // ����Ϊֻ��
    ui.textEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding); // ����Ӧ��С
    ui.textEdit->setLineWrapMode(QTextEdit::WidgetWidth);  // �Զ�����

    // �� QTextEdit ��ӵ� drawingArea
    QVBoxLayout* layout = new QVBoxLayout(ui.drawingArea);
    layout->addWidget(ui.textEdit);
    ui.drawingArea->setLayout(layout);

    // ����Ĭ��ѡ��Ϊ ID3
    ui.comboBox->setCurrentIndex(0);  // 0��ʾ ID3

    // ���� QComboBox ���źŲ�
    connect(ui.comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onComboBoxChanged(int)));

    connect(ui.chooseSampleButton, &QPushButton::clicked, this, &DecisionTreeForDating::chooseSample);
    connect(ui.chooseInputButton, &QPushButton::clicked, this, &DecisionTreeForDating::chooseInput);
    connect(ui.beginPredictButton, &QPushButton::clicked, this, &DecisionTreeForDating::beginPredict);

}

DecisionTreeForDating::~DecisionTreeForDating()
{}

// ���� QComboBox ��ѡ��ı�
void DecisionTreeForDating::onComboBoxChanged(int index) {
    // ���� QComboBox ��ѡ������ useC45
    if (index == 0) {
        // ���ѡ���� ID3
        decisionTree.useC45 = false;  // ʹ�� ID3
    }
    else if (index == 1) {
        // ���ѡ���� C4.5
        decisionTree.useC45 = true;   // ʹ�� C4.5
    }
}

void DecisionTreeForDating::chooseSample() {
    // ���ļ��Ի���ѡ��CSV�ļ�
    QString fileName = QFileDialog::getOpenFileName(this, QStringLiteral("ѡ����������"), "", "CSV Files (*.csv)");
    if (fileName.isEmpty()) {
        return;
    }

    QFileInfo fileInfo(fileName);
    ui.lineEditSample->setText(fileInfo.fileName());

    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::critical(this, QStringLiteral("����"), QStringLiteral("�޷����ļ���"));
        return;
    }

    QTextStream in(&file);
    vector<DataPoint> sampleData;

    // ��ȡCSV�ļ�
    while (!in.atEnd()) {
        QString line = in.readLine();
        QStringList columns = line.split(",");

        if (columns.size() != 5) {
            continue;  // ����ÿ����5���ֶΣ�weather, temperature, humidity, wind, label
        }

        DataPoint point;
        point.weather = columns[0].toStdString();
        point.temperature = columns[1].toStdString();
        point.humidity = columns[2].toStdString();
        point.wind = columns[3].toStdString();
        point.date = columns[4].toStdString();  // ��ǩ

        sampleData.push_back(point);
    }

    // ����������������ѵ��
    decisionTree.trainData = sampleData;

    // ����Ƿ������������ݺʹ�Ԥ������
    if (decisionTree.trainData.empty()) {
        QMessageBox::warning(this, QStringLiteral("����"), QStringLiteral("��������Ϊ�գ�"));
        return;
    }
}

void DecisionTreeForDating::chooseInput() {
    // ���ļ��Ի���ѡ���Ԥ������CSV�ļ�
    QString fileName = QFileDialog::getOpenFileName(this, QStringLiteral("ѡ���Ԥ������"), "", "CSV Files (*.csv)");
    if (fileName.isEmpty()) {
        return;
    }

    QFileInfo fileInfo(fileName);
    ui.lineEditInput->setText(fileInfo.fileName());

    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::critical(this, QStringLiteral("����"), QStringLiteral("�޷����ļ���"));
        return;
    }

    QTextStream in(&file);
    vector<DataPoint> testData;

    // ��ȡCSV�ļ�
    while (!in.atEnd()) {
        QString line = in.readLine();
        QStringList columns = line.split(",");

        if (columns.size() != 4) {
            continue;  // ����ÿ����4���ֶΣ�weather, temperature, humidity, wind
        }

        DataPoint point;
        point.weather = columns[0].toStdString();
        point.temperature = columns[1].toStdString();
        point.humidity = columns[2].toStdString();
        point.wind = columns[3].toStdString();

        testData.push_back(point);
    }

    // �����Ԥ������
    decisionTree.testData = testData;

    // ����Ƿ������������ݺʹ�Ԥ������
    if (decisionTree.testData.empty()) {
        QMessageBox::warning(this, "Error", "Test data is empty.");
        return;
    }
}

void DecisionTreeForDating::beginPredict() {
    // ��ȡѡ�е��㷨����
    QString selectedAlgorithm = ui.comboBox->currentText();

    ui.drawingArea->update();

    // ����Ƿ������������ݺʹ�Ԥ������
    if (decisionTree.trainData.empty() || decisionTree.testData.empty()) {
        QMessageBox::warning(this, "����", QStringLiteral("�������ݻ��Ԥ������Ϊ�գ�"));
        return;
    }

    // ������������ѵ��
    vector<string> features = { "weather", "temperature", "humidity", "wind" };

    // �ж��Ƿ�ʹ����Ϣ������
    if (selectedAlgorithm == "C4.5") decisionTree.useC45 == true;

    root = decisionTree.buildTree(decisionTree.trainData, features);  // ����ȫ�ָ��ڵ�

    if (!root) {
        QMessageBox::warning(this, QStringLiteral("����"), QStringLiteral("������δ�ܳɹ�������"));
        return;
    }

    // ���ɾ������ı������� QTextEdit
    QString treeText = generateTreeText(root, 0, true);
    ui.textEdit->setPlainText(treeText);  // ���ı����õ� QTextEdit

    // ����һ���ַ������������е�Ԥ����
    QString results;

    // ����ѡ����㷨����Ԥ��
    for (const auto& test : decisionTree.testData) {
        string result;
        if (selectedAlgorithm == "ID3") {
            result = decisionTree.predictID3(root, test);
        }
        else if (selectedAlgorithm == "C4.5") {
            result = decisionTree.predictC45(root, test);
        }

        // ��Ԥ������ӵ��ַ�����
        results += QString::fromStdString(result) + "\n";
    }

    // ���Ԥ��������Ϣ��
    QMessageBox::information(this, QStringLiteral("Ԥ������"), results);
}

QString DecisionTreeForDating::generateTreeText(TreeNode* node, int level, bool isLast) {
    if (!node) return "";

    QString text;
    QString indent(level * 4, ' '); // �����ı�����
    QString treeSymbol = isLast ? "    " : "|   "; // ���Ƶ�ǰ��ε����ӷ���

    // ��ӵ�ǰ�ڵ���Ϣ
    text += indent + (level > 0 ? treeSymbol : "") + QString::fromStdString(node->label.empty() ? node->feature : node->label) + "\n";

    // �����ӽڵ�
    for (auto it = node->children.begin(); it != node->children.end(); ++it) {
        bool lastChild = (std::next(it) == node->children.end()); // �ж��Ƿ�Ϊ���һ���ӽڵ�
        text += generateTreeText(it->second, level + 1, lastChild); // �ݹ�
    }
    return text;
}


