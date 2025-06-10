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

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 自定义数据集类
class Mydataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


# VAER模型类
class VAER(nn.Module):
    def __init__(self, input_dim, seq_len, latent_dim, output_dim):
        super(VAER, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # 编码器网络
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=1),
            # nn.Linear(input_dim,64),
            # nn.GELU(),
            # nn.Conv1d(64, 64, kernel_size=1),
            # nn.GELU(),
        )
        self.mu_layer = nn.Conv1d(64, latent_dim, kernel_size=1)
        self.logvar_layer = nn.Conv1d(64, latent_dim, kernel_size=1)
        # 解码器网络
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, 64, kernel_size=1),
            # nn.Linear(latent_dim,64),
            # nn.GELU(),
            # nn.Conv1d(64, 64, kernel_size=1),
            # nn.GELU(),
            nn.Conv1d(64, input_dim, kernel_size=1),
            # nn.Linear(64,input_dim)
        )

        # 回归预测模块
        self.regression = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(latent_dim, 128),
            nn.GELU(),  ##
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp( 0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 编码过程
        enc_out = self.encoder(x)
        mu = self.mu_layer(enc_out)
        logvar = self.logvar_layer(enc_out)

        # 重参数化
        # z = self.reparameterize(mu, logvar)
        z = enc_out
        # 数据重建
        # x_recon = self.decoder(z)
        x_recon = self.decoder(z)

        # 回归预测
        y_pred = self.regression(z)

        return x_recon, y_pred, mu, logvar


# 自定义损失函数
def loss_function(x_recon, x, y_pred, y, mu, logvar):
    # 重建损失
    recon_loss = F.mse_loss(x_recon, x)

    # KL散度损失
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # 回归损失
    reg_loss = F.mse_loss(y_pred, y.view(-1, 1))

    # 总损失
    total_loss = recon_loss + kl_loss + reg_loss
    return total_loss


# 可视化函数（仅显示）
def plot_comparison(y_true, y_pred):
    plt.figure(figsize=(12, 6))

    # 折线对比图
    # plt.subplot(1, 2, 1)
    plt.plot(y_true[:], label='True Values')
    plt.plot(y_pred[:], label='Predicted Values')
    plt.title('VAER Prediction Comparison ')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    plt.close()
    # # 散点相关图
    # plt.subplot(1, 2, 2)
    # plt.scatter(y_true, y_pred, alpha=0.5)
    # plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--')
    # plt.xlabel('True Values')
    # plt.ylabel('Predicted Values')
    # plt.title(f'Correlation (R²={r2_score(y_true, y_pred):.3f})')
    # plt.grid(True)


# 模型训练函数（GPU支持）
def train_model(model, train_loader, val_loader, epochs, lr):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.permute(0, 2, 1).to(device)  # 转移到GPU
            y = y.to(device)

            optimizer.zero_grad()
            x_recon, y_pred, mu, logvar = model(x)
            loss = loss_function(x_recon, x, y_pred, y, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.permute(0, 2, 1).to(device)
                y = y.to(device)
                x_recon, y_pred, mu, logvar = model(x)
                loss = loss_function(x_recon, x, y_pred, y, mu, logvar)
                val_loss += loss.item()

        # 保存最佳模型
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        # 打印训练信息
        print(f'Epoch {epoch + 1}/{epochs} | '
              f'Train Loss: {train_loss / len(train_loader):.4f} | '
              f'Val Loss: {avg_val_loss:.4f}')


# 模型评估函数（GPU支持）
def evaluate_model(model, test_loader):
    model = model.to(device)
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.permute(0, 2, 1).to(device)
            y = y.to(device)

            _, pred, _, _ = model(x)
            y_true.append(y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true).flatten()
    y_pred = np.concatenate(y_pred).flatten()

    print('\nEvaluation Results:')
    print(f'R² Score: {r2_score(y_true, y_pred):.4f}')
    print(f'RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}')
    print(f'MAE: {mean_absolute_error(y_true, y_pred):.4f}')

    plot_comparison(y_true, y_pred)


# 主程序
if __name__ == "__main__":
    # 数据加载与预处理
    data = pd.read_excel(r'C:\Python\DS\pta.xlsx', engine='openpyxl')
    data = data.values
    print(data.shape)
    TRAIN_SIZE = 182
    x_temp = data[:, 0:-1]
    y_temp = data[:, -1]
    x_train = x_temp[:TRAIN_SIZE,:]
    y_train = y_temp[:TRAIN_SIZE,]
    x_test = x_temp[TRAIN_SIZE:,:]
    y_test = y_temp[TRAIN_SIZE:]
    # 从训练集划分验证集
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )

    # 数据标准化
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).reshape(-1, 1, x_train.shape[1])
    x_val = scaler.transform(x_val).reshape(-1, 1, x_val.shape[1])
    x_test = scaler.transform(x_test).reshape(-1, 1, x_test.shape[1])

    # 创建数据加载器
    train_dataset = Mydataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    val_dataset = Mydataset(torch.FloatTensor(x_val), torch.FloatTensor(y_val))
    test_dataset = Mydataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # 模型初始化
    model = VAER(
        input_dim=x_train.shape[-1],
        seq_len=1,
        latent_dim=64,
        output_dim=1
    ).to(device)

    # 训练模型
    print("Starting Training...")
    train_model(model, train_loader, val_loader, epochs=200, lr=0.001)

    # 加载最佳模型进行评估
    print("\nLoading Best Model for Evaluation...")
    model.load_state_dict(
        torch.load('best_model.pth',
                   map_location=device,
                   weights_only=True)
    )
    evaluate_model(model, test_loader)