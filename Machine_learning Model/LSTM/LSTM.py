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


# LSTM模型类
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, proj_dim=128):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 输入投影层（与Transformer相同的结构）
        self.input_proj = nn.Linear(input_dim, proj_dim)

        # LSTM层
        self.lstm = nn.LSTM(proj_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=0.1 if num_layers > 1 else 0)

        # 回归预测层（与Transformer相同的结构）
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # 初始化解隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)

        # 输入投影
        x = self.input_proj(x)

        # LSTM处理
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出
        out = out[:, -1, :]

        # 回归预测
        return self.regressor(out)


# 可视化函数
def plot_comparison(y_true, y_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='True Values')
    plt.plot(y_pred, label='Predicted Values')
    plt.title('LSTM Prediction Comparison')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    plt.close()


# 模型训练函数
def train_model(model, train_loader, val_loader, epochs, lr):
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
            torch.save(model.state_dict(), 'best_lstm.pth')

        print(f'Epoch {epoch + 1}/{epochs} | '
              f'Train Loss: {train_loss / len(train_loader):.4f} | '
              f'Val Loss: {avg_val_loss:.4f}')


# 模型评估函数
def evaluate_model(model, test_loader):
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

    print('\nLSTM Evaluation Results:')
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

    # 初始化LSTM模型
    model = LSTMModel(
        input_dim=1,  # 每个时间步的特征维度
        hidden_dim=128,  # LSTM隐藏单元维度
        num_layers=3,  # LSTM层数
        output_dim=1  # 输出维度
    ).to(device)

    # 训练模型
    print("Starting LSTM Training...")
    train_model(model, train_loader, val_loader, epochs=200, lr=0.001)

    # 加载最佳模型进行评估
    print("\nLoading Best LSTM Model...")
    model.load_state_dict(torch.load('best_lstm.pth', weights_only=True))
    evaluate_model(model, test_loader)