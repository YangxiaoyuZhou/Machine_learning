import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")


# 宽度学习系统类
class BroadLearningSystem:
    def __init__(self, n_feature_nodes=10, n_window_nodes=10, n_enhance_nodes=100, C=2 ** -30):
        """
        初始化BLS模型参数

        参数:
        n_feature_nodes: 特征节点数量
        n_window_nodes: 窗口节点数量
        n_enhance_nodes: 增强节点数量
        C: 正则化参数
        """
        self.n_feature_nodes = n_feature_nodes
        self.n_window_nodes = n_window_nodes
        self.n_enhance_nodes = n_enhance_nodes
        self.C = C
        self.Wh = None
        self.beta = None
        self.feature_weights = None
        self.enhance_weights = None
        self.scaler = StandardScaler()

    def _sparse_autoencoder(self, X, n_nodes):
        """生成稀疏自编码器权重"""
        L = 0.5  # 稀疏参数
        input_size = X.shape[1]

        # 随机初始化权重
        W = np.random.normal(0, 1, (input_size, n_nodes))
        b = np.random.normal(0, 1, (1, n_nodes))

        # 线性变换
        Z = np.dot(X, W) + b

        # 稀疏约束
        sparse_constraint = L * np.sign(Z) * np.maximum(0, np.abs(Z) - L)
        Z = Z - sparse_constraint

        return W, b, Z

    def fit(self, X, y):
        """训练BLS模型"""
        # 数据标准化
        X = self.scaler.fit_transform(X)

        # 特征节点生成
        self.feature_weights = []
        feature_nodes = []

        for _ in range(self.n_feature_nodes):
            W, b, Z = self._sparse_autoencoder(X, self.n_window_nodes)
            self.feature_weights.append((W, b))
            feature_nodes.append(Z)

        # 拼接所有特征节点
        Z = np.hstack(feature_nodes)

        # 增强节点生成
        self.enhance_weights = []
        enhance_nodes = []

        for _ in range(self.n_enhance_nodes):
            W, b, H = self._sparse_autoencoder(Z, 1)
            self.enhance_weights.append((W, b))
            enhance_nodes.append(H)

        # 拼接所有增强节点
        H = np.hstack(enhance_nodes)

        # 组合特征节点和增强节点
        A = np.hstack([Z, H])

        # 计算输出权重
        if A.shape[0] < A.shape[1]:
            # 样本数小于特征数时使用岭回归
            self.beta = np.dot(np.dot(A.T, np.linalg.inv(np.dot(A, A.T) +
                                                         np.eye(A.shape[0]) * self.C)), y)
        else:
            # 样本数大于特征数时使用伪逆
            self.beta = np.dot(np.linalg.pinv(A), y)

    def predict(self, X):
        """使用训练好的模型进行预测"""
        # 数据标准化
        X = self.scaler.transform(X)

        # 特征节点生成
        feature_nodes = []
        for W, b in self.feature_weights:
            Z = np.dot(X, W) + b
            feature_nodes.append(Z)

        # 拼接所有特征节点
        Z = np.hstack(feature_nodes)

        # 增强节点生成
        enhance_nodes = []
        for W, b in self.enhance_weights:
            H = np.dot(Z, W) + b
            enhance_nodes.append(H)

        # 拼接所有增强节点
        H = np.hstack(enhance_nodes)

        # 组合特征节点和增强节点
        A = np.hstack([Z, H])

        # 预测
        return np.dot(A, self.beta)


# 可视化函数
def plot_comparison(y_true, y_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='True Values')
    plt.plot(y_pred, label='Predicted Values')
    plt.title('BLS Prediction Comparison')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    plt.close()


# 模型评估函数
def evaluate_model(y_true, y_pred):
    print('\nBLS Evaluation Results:')
    print(f'R² Score: {r2_score(y_true, y_pred):.4f}')
    print(f'RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}')
    print(f'MAE: {mean_absolute_error(y_true, y_pred):.4f}')

    plot_comparison(y_true, y_pred)


# 主程序
if __name__ == "__main__":
    # 数据加载与预处理
    pp = pd.read_csv(r"C:\Python\DS\gt_2015.csv").values

    # 数据划分
    x = pp[:, 1:-2].astype(np.float32)
    y = pp[:, -1].astype(np.float32)

    # 划分数据集
    x_train = x[:6000, :]
    y_train = y[:6000]
    x_test = x[6000:7500, :]
    y_test = y[6000:7500]

    # 验证集划分
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42
    )

    # 初始化BLS模型
    print("Initializing Broad Learning System...")
    bls = BroadLearningSystem(
        n_feature_nodes=20,  # 特征节点数量
        n_window_nodes=15,  # 窗口节点数量
        n_enhance_nodes=500,  # 增强节点数量
        C=2 ** -30  # 正则化参数
    )

    # 训练模型
    print("Training BLS Model...")
    bls.fit(x_train, y_train)

    # 验证集评估
    print("\nEvaluating on Validation Set...")
    y_val_pred = bls.predict(x_val)
    evaluate_model(y_val, y_val_pred)

    # 测试集评估
    print("\nEvaluating on Test Set...")
    y_test_pred = bls.predict(x_test)
    evaluate_model(y_test, y_test_pred)