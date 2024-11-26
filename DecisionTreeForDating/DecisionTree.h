#pragma once
#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <string>
#include <algorithm>

using namespace std;

// ���ڴ洢������Ŀ
struct DataPoint {
    string weather;
    string temperature;
    string humidity;
    string wind;
    string date; // ��ǩ��Լ�����
};

// �������ڵ�
struct TreeNode {
    string feature;          // ��ǰ�ڵ������
    map<string, TreeNode*> children; // �ӽڵ㣨��������ֵ��
    string label;            // �����Ҷ�ڵ㣬��洢�����ǩ
};

class DecisionTree {
public:
    DecisionTree(bool useC45 = false);  
    TreeNode* buildTree(const vector<DataPoint>& data, vector<string>& features);

    // ʹ�� ID3 ����Ԥ��
    string predictID3(TreeNode* root, const DataPoint& test);

    // ʹ�� C4.5 ����Ԥ��
    string predictC45(TreeNode* root, const DataPoint& test);

    bool useC45;  // �Ƿ�ʹ��C4.5

    // ���ڴ洢ѵ�����ݺͲ�������
    vector<DataPoint> trainData;
    vector<DataPoint> testData;
private:
    double entropy(const vector<DataPoint>& data);
    double informationGain(const vector<DataPoint>& data, const string& feature);
    double gainRatio(const vector<DataPoint>& data, const string& feature);
};

#endif // DECISIONTREE_H


