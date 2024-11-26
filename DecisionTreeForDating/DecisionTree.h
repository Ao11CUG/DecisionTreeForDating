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

// 用于存储数据条目
struct DataPoint {
    string weather;
    string temperature;
    string humidity;
    string wind;
    string date; // 标签（约会与否）
};

// 决策树节点
struct TreeNode {
    string feature;          // 当前节点的特征
    map<string, TreeNode*> children; // 子节点（键是特征值）
    string label;            // 如果是叶节点，则存储分类标签
};

class DecisionTree {
public:
    DecisionTree(bool useC45 = false);  
    TreeNode* buildTree(const vector<DataPoint>& data, vector<string>& features);

    // 使用 ID3 进行预测
    string predictID3(TreeNode* root, const DataPoint& test);

    // 使用 C4.5 进行预测
    string predictC45(TreeNode* root, const DataPoint& test);

    bool useC45;  // 是否使用C4.5

    // 用于存储训练数据和测试数据
    vector<DataPoint> trainData;
    vector<DataPoint> testData;
private:
    double entropy(const vector<DataPoint>& data);
    double informationGain(const vector<DataPoint>& data, const string& feature);
    double gainRatio(const vector<DataPoint>& data, const string& feature);
};

#endif // DECISIONTREE_H


