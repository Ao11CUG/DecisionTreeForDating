#include "DecisionTree.h"

DecisionTree::DecisionTree(bool useC45) : useC45(useC45) {
}

// 计算熵
double DecisionTree::entropy(const vector<DataPoint>& data) {
    map<string, int> labelCount;
    for (const auto& d : data) {
        labelCount[d.date]++;
    }

    double entropy = 0.0;
    int total = data.size();
    for (const auto& pair : labelCount) {
        double prob = (double)pair.second / total;
        entropy -= prob * log2(prob);
    }

    return entropy;
}

// 计算信息增益
double DecisionTree::informationGain(const vector<DataPoint>& data, const string& feature) {
    map<string, vector<DataPoint>> subsets;
    for (const auto& d : data) {
        if (feature == "weather") {
            subsets[d.weather].push_back(d);
        }
        else if (feature == "temperature") {
            subsets[d.temperature].push_back(d);
        }
        else if (feature == "humidity") {
            subsets[d.humidity].push_back(d);
        }
        else if (feature == "wind") {
            subsets[d.wind].push_back(d);
        }
    }

    double totalEntropy = entropy(data);
    double weightedEntropy = 0.0;
    int total = data.size();

    for (const auto& pair : subsets) {
        double subsetEntropy = entropy(pair.second);
        weightedEntropy += ((double)pair.second.size() / total) * subsetEntropy;
    }

    return totalEntropy - weightedEntropy;
}

// 计算信息增益率
double DecisionTree::gainRatio(const vector<DataPoint>& data, const string& feature) {
    double infoGain = informationGain(data, feature);

    map<string, vector<DataPoint>> subsets;
    for (const auto& d : data) {
        if (feature == "weather") {
            subsets[d.weather].push_back(d);
        }
        else if (feature == "temperature") {
            subsets[d.temperature].push_back(d);
        }
        else if (feature == "humidity") {
            subsets[d.humidity].push_back(d);
        }
        else if (feature == "wind") {
            subsets[d.wind].push_back(d);
        }
    }

    double splitInfo = 0.0;
    int total = data.size();
    for (const auto& pair : subsets) {
        double prob = (double)pair.second.size() / total;
        splitInfo -= prob * log2(prob);
    }

    return (splitInfo != 0) ? infoGain / splitInfo : 0.0;
}

// 构建决策树
TreeNode* DecisionTree::buildTree(const vector<DataPoint>& data, vector<string>& features) {
    if (data.empty()) return nullptr;

    // 如果数据纯净（只有一个类别），返回叶节点
    map<string, int> labelCount;
    for (const auto& d : data) {
        labelCount[d.date]++;
    }
    if (labelCount.size() == 1) {
        TreeNode* leaf = new TreeNode();
        leaf->label = labelCount.begin()->first;
        return leaf;
    }

    if (features.empty()) {
        TreeNode* leaf = new TreeNode();
        leaf->label = max_element(labelCount.begin(), labelCount.end(), [](const pair<string, int>& a, const pair<string, int>& b) {
            return a.second < b.second;
            })->first;
        return leaf;
    }

    // 选择最优特征
    string bestFeature;
    double bestGain = -1;
    for (const auto& feature : features) {
        double gain = useC45 ? gainRatio(data, feature) : informationGain(data, feature);
        if (gain > bestGain) {
            bestGain = gain;
            bestFeature = feature;
        }
    }

    // 创建当前节点
    TreeNode* node = new TreeNode();
    node->feature = bestFeature;

    // 按特征值分割数据
    map<string, vector<DataPoint>> subsets;
    for (const auto& d : data) {
        if (bestFeature == "weather") {
            subsets[d.weather].push_back(d);
        }
        else if (bestFeature == "temperature") {
            subsets[d.temperature].push_back(d);
        }
        else if (bestFeature == "humidity") {
            subsets[d.humidity].push_back(d);
        }
        else if (bestFeature == "wind") {
            subsets[d.wind].push_back(d);
        }
    }

    // 递归构建子节点
    vector<string> remainingFeatures = features;
    remainingFeatures.erase(remove(remainingFeatures.begin(), remainingFeatures.end(), bestFeature), remainingFeatures.end());
    for (const auto& pair : subsets) {
        node->children[pair.first] = buildTree(pair.second, remainingFeatures);
    }

    return node;
}

// 使用 ID3 进行预测
string DecisionTree::predictID3(TreeNode * root, const DataPoint & test) {
    if (root->label != "") {
        return root->label;
    }

    if (root->children.find(test.weather) != root->children.end()) {
        return predictID3(root->children[test.weather], test);
    }
    else if (root->children.find(test.temperature) != root->children.end()) {
        return predictID3(root->children[test.temperature], test);
    }
    else if (root->children.find(test.humidity) != root->children.end()) {
        return predictID3(root->children[test.humidity], test);
    }
    else if (root->children.find(test.wind) != root->children.end()) {
        return predictID3(root->children[test.wind], test);
    }

    return "";
}

// 使用 C4.5 进行预测
string DecisionTree::predictC45(TreeNode* root, const DataPoint& test) {
    if (root->label != "") {
        return root->label;
    }

    if (root->children.find(test.weather) != root->children.end()) {
        return predictC45(root->children[test.weather], test);
    }
    else if (root->children.find(test.temperature) != root->children.end()) {
        return predictC45(root->children[test.temperature], test);
    }
    else if (root->children.find(test.humidity) != root->children.end()) {
        return predictC45(root->children[test.humidity], test);
    }
    else if (root->children.find(test.wind) != root->children.end()) {
        return predictC45(root->children[test.wind], test);
    }

    return "";
}



