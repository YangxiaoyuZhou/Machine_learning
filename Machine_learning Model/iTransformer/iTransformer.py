import torch
import torch.nn as nn
import torch.nn.functional as F
from holoviews.plotting.bokeh.styles import alpha
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 配置类
class Config:
    def __init__(self, feature_dim):
        self.enc_in = 1  # 输入特征维度
        self.seq_len = feature_dim  # 序列长度（特征数量）
        self.pred_len = 1  # 预测长度
        self.d_model = 64  # 模型维度
        self.n_heads = 4  # 注意力头数
        self.e_layers = 2  # 编码器层数
        self.d_ff = 256  # 前馈网络维度
        self.dropout = 0.1  # Dropout概率
        self.activation = 'gelu'  # 激活函数


# 数据嵌入层
class DataEmbedding(nn.Module):
    def __init__(self, seq_len, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding
        return self.dropout(x)


# iTransformer模型
class iTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 嵌入层
        self.enc_embedding = DataEmbedding(
            seq_len=config.seq_len,
            c_in=config.enc_in,
            d_model=config.d_model,
            dropout=config.dropout
        )

        # Transformer编码器
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_ff,
                dropout=config.dropout,
                activation=config.activation,
                batch_first=True
            ),
            num_layers=config.e_layers
        )

        # 回归预测头
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.seq_len * config.d_model, 128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.pred_len),
            nn.Flatten(start_dim=0)  # 新增扁平化层
        )

    def forward(self, x):
        # 输入验证 [batch, seq_len, 1]
        assert x.dim() == 3, f"输入维度错误，应为3D，得到{x.dim()}D"

        # 标准化
        means = x.mean(dim=1, keepdim=True)
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # 嵌入处理
        enc_out = self.enc_embedding(x)  # [batch, seq_len, d_model]

        # 编码处理
        enc_out = self.encoder(enc_out)  # [batch, seq_len, d_model]

        # 预测
        dec_out = self.projection(enc_out)  # [batch]
        return dec_out.squeeze()  # 确保输出维度为 [batch]


# 数据集类
class TSRegressionDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 训练函数
def train_model(model, train_loader, val_loader, epochs=200, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device).unsqueeze(-1)  # [batch, seq_len, 1]
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            # 维度验证
            assert pred.shape == y.shape, f"预测形状{pred.shape} vs 目标{y.shape}"
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device).unsqueeze(-1)
                y = y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), 'best_itransformer.pth')

        print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}')


# 评估函数
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device).unsqueeze(-1)
            pred = model(x).cpu().numpy().flatten()  # 确保一维输出
            y_true.extend(y.numpy().flatten())
            y_pred.extend(pred)

    # 转换为numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 形状验证
    assert y_true.shape == y_pred.shape, f"形状不匹配：真实值{len(y_true)} vs 预测值{len(y_pred)}"

    print("\nEvaluation Results:")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")

    # 可视化
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:], label='True')
    plt.plot(y_pred[:], label='Predicted')
    plt.title("iTransformer Prediction Comparison")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    # 加载数据
    data = pd.read_csv(r"C:\Python\DS\gt_2014.csv").values

    # 数据划分
    x = data[:, 1:-2].astype(np.float32)
    y = data[:, -1].astype(np.float32)

    # 数据集划分
    x_train, x_test = x[1500:7500], x[:1500]
    y_train, y_test = y[1500:7500], y[:1500]
    # x_train = x[:6000, :]
    # y_train = y[:6000]
    # x_test = x[6000:7500, :]
    # y_test = y[6000:7500]

    # 标准化
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 验证集划分
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )

    # 创建配置
    feature_dim = x_train.shape[1]
    config = Config(feature_dim)

    # 创建数据集（确保三维形状）
    train_dataset = TSRegressionDataset(
        x_train.reshape(-1, feature_dim),  # [样本数, 特征数]
        y_train
    )
    val_dataset = TSRegressionDataset(
        x_val.reshape(-1, feature_dim),
        y_val
    )
    test_dataset = TSRegressionDataset(
        x_test.reshape(-1, feature_dim),
        y_test
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # 初始化模型
    model = iTransformer(config)

    # 训练
    print("Starting Training...")
    train_model(model, train_loader, val_loader)

    # 评估
    print("\nEvaluating Best Model...")
    model.load_state_dict(torch.load('best_itransformer.pth', map_location=device, weights_only=True))
    evaluate_model(model, test_loader)