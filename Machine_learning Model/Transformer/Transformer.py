import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore",
    message=".*Torch was not compiled with flash attention.*",
    category=UserWarning)

# 自定义数据集类
class Mydataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


# Transformer模型类
class TimeSeriesTransformer(nn.Module):
    def __init__(self, feature_dim, d_model, nhead, num_layers, output_dim):
        super(TimeSeriesTransformer, self).__init__()

        # 输入投影层
        self.input_proj = nn.Linear(feature_dim, d_model)  # 特征维度映射

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 回归预测
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.regressor(x)


# 位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x形状: [batch, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


# 可视化函数
def plot_comparison(y_true, y_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='True Values')
    plt.plot(y_pred, label='Predicted Values')
    plt.title('iTransformer Prediction Comparison')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    plt.close()


# 模型训练函数
def train_transformer(model, train_loader, val_loader, epochs, lr):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss = float('inf')

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)  # [batch, seq_len, feature_dim]
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred.squeeze(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred.squeeze(), y)
                val_loss += loss.item()

        # 保存最佳模型
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_transformer.pth')

        print(f'Epoch {epoch + 1}/{epochs} | '
              f'Train Loss: {train_loss / len(train_loader):.4f} | '
              f'Val Loss: {avg_val_loss:.4f}')


# 模型评估函数
def evaluate_transformer(model, test_loader):
    model = model.to(device)
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            y_true.append(y.cpu().numpy())
            y_pred.append(pred.squeeze().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    print('\nEvaluation Results:')
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
    # x_train = x[1500:7500, :]
    # y_train = y[1500:7500]
    # x_test = x[:1500, :]
    # y_test = y[:1500]

    # 验证集划分
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1


        , random_state=42
    )

    # 数据标准化
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    # 转换为三维张量 [samples, seq_len, features]
    x_train = x_train.reshape(-1, x_train.shape[1], 1)
    x_val = x_val.reshape(-1, x_val.shape[1], 1)
    x_test = x_test.reshape(-1, x_test.shape[1], 1)

    # 创建数据加载器
    train_dataset = Mydataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    val_dataset = Mydataset(torch.FloatTensor(x_val), torch.FloatTensor(y_val))
    test_dataset = Mydataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # 初始化模型参数（关键修改）
    model = TimeSeriesTransformer(
        feature_dim=1,  # 每个时间步的特征维度
        d_model=128,
        nhead=4,
        num_layers=3,
        output_dim=1
    ).to(device)

    # 训练模型
    print("Starting Transformer Training...")
    train_transformer(model, train_loader, val_loader, epochs=200, lr=0.001)

    # 加载最佳模型进行评估
    print("\nLoading Best Transformer Model...")
    # model.load_state_dict(torch.load('best_transformer.pth', map_location=device, weights_only=True))
    evaluate_transformer(model, test_loader)