import torch
import numpy as np


class DecisionNode:
    """决策树节点类"""

    def __init__(self, feature_idx=None, threshold=None, value=None, left=None, right=None):
        self.feature_idx = feature_idx  # 分裂特征索引
        self.threshold = threshold  # 分裂阈值
        self.value = value  # 叶节点值（类别分布）
        self.left = left  # 左子树
        self.right = right  # 右子树


class DecisionTreeClassifier:
    """决策树分类器"""

    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth  # 最大深度
        self.min_samples_split = min_samples_split  # 最小分裂样本数
        self.root = None  # 根节点

    def _gini(self, y):
        """计算基尼不纯度"""
        if y.size(0) == 0:
            return 0
        p = torch.bincount(y) / y.size(0)
        return 1 - torch.sum(p ** 2)

    def _best_split(self, X, y):
        """寻找最佳分裂特征和阈值"""
        best_gini = float('inf')
        best_feature, best_threshold = None, None

        for feature_idx in range(X.shape[1]):
            thresholds = torch.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_idx = X[:, feature_idx] <= threshold
                y_left = y[left_idx]
                y_right = y[~left_idx]

                gini = (y_left.size(0) * self._gini(y_left) +
                        y_right.size(0) * self._gini(y_right)) / y.size(0)

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """递归构建决策树"""
        # 停止条件
        if (self.max_depth is not None and depth >= self.max_depth) or \
                y.size(0) < self.min_samples_split or \
                len(torch.unique(y)) == 1:
            value = torch.bincount(y, minlength=self.n_classes) / y.size(0)
            return DecisionNode(value=value)

        # 寻找最佳分裂
        feature_idx, threshold = self._best_split(X, y)
        if feature_idx is None:
            value = torch.bincount(y, minlength=self.n_classes) / y.size(0)
            return DecisionNode(value=value)

        # 分裂数据
        left_idx = X[:, feature_idx] <= threshold
        X_left, y_left = X[left_idx], y[left_idx]
        X_right, y_right = X[~left_idx], y[~left_idx]

        # 递归构建子树
        left = self._build_tree(X_left, y_left, depth + 1)
        right = self._build_tree(X_right, y_right, depth + 1)
        return DecisionNode(feature_idx, threshold, left=left, right=right)

    def fit(self, X, y):
        """训练决策树"""
        self.n_classes = len(torch.unique(y))
        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.long)
        self.root = self._build_tree(X, y)

    def _predict_sample(self, x, node):
        """预测单个样本"""
        if node.value is not None:
            return node.value.argmax()
        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def predict(self, X):
        """预测"""
        X = torch.as_tensor(X, dtype=torch.float32)
        return torch.tensor([self._predict_sample(x, self.root) for x in X])


# 示例使用
if __name__ == "__main__":
    # 创建示例数据
    X = torch.tensor([[2.0, 1.5], [1.5, 2.0], [1.0, 1.0],
                      [3.0, 3.0], [2.5, 2.5], [3.5, 3.0]])
    y = torch.tensor([0, 1, 2, 3, 4, 5])

    # 训练决策树
    tree = DecisionTreeClassifier(max_depth=2)
    tree.fit(X, y)

    # 预测新样本
    test_sample = torch.tensor([[2.0, 2.0]])
    print("Prediction:", tree.predict(test_sample).item())