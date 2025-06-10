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
import os

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore",
                        message=".*Torch was not compiled with flash attention.*",
                        category=UserWarning)


# 自定义数据集类（保持与原始代码相同）
class Mydataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


# 生成器 (Generator) - 使用LSTM架构
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 输入投影层（与原始Transformer相同）
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # LSTM层
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=0.1 if num_layers > 1 else 0)

        # 输出层（生成预测值）
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x, noise=None):
        # 如果需要生成样本，添加噪声
        if noise is not None:
            x = x + noise

        # 输入投影
        x = self.input_proj(x)

        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)

        # LSTM处理
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出
        out = out[:, -1, :]

        # 回归预测
        return self.output_layer(out)


# 判别器 (Discriminator)
class Discriminator(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers, output_dim=1):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 为特征和数据创建联合输入
        self.input_proj = nn.Linear(feature_dim + output_dim, hidden_dim)

        # LSTM层
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=0.1 if num_layers > 1 else 0)

        # 分类层（判断真实/生成）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        # 将特征与预测值连接
        y_reshaped = y.view(y.size(0), 1, -1).repeat(1, x.size(1), 1)
        combined = torch.cat((x, y_reshaped), dim=2)

        # 输入投影
        combined = self.input_proj(combined)

        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, combined.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, combined.size(0), self.hidden_dim).to(device)

        # LSTM处理
        out, _ = self.lstm(combined, (h0, c0))

        # 取最后一个时间步的输出
        out = out[:, -1, :]

        # 分类判断
        return self.classifier(out)


# GAN模型整合
class GANModel:
    def __init__(self, generator, discriminator, gen_optimizer, disc_optimizer):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.criterion = nn.BCELoss()
        self.mse = nn.MSELoss()

    def train_step(self, x, y_true):
        batch_size = x.size(0)

        # 创建真实和假的标签
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        ############################
        # 训练判别器 (Discriminator)
        ############################
        self.disc_optimizer.zero_grad()

        # 1. 真实样本的判别
        real_output = self.discriminator(x, y_true)
        d_loss_real = self.criterion(real_output, real_labels)

        # 2. 生成样本的判别
        y_fake = self.generator(x)
        fake_output = self.discriminator(x, y_fake.detach())
        d_loss_fake = self.criterion(fake_output, fake_labels)

        # 判别器总损失
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.disc_optimizer.step()

        ############################
        # 训练生成器 (Generator)
        ############################
        self.gen_optimizer.zero_grad()

        # 1. 对抗损失：尝试让判别器认为生成样本是真实的
        fake_output_gen = self.discriminator(x, y_fake)
        g_loss_adv = self.criterion(fake_output_gen, real_labels)

        # 2. 内容损失：使生成值接近真实值
        g_loss_content = self.mse(y_fake.squeeze(), y_true)

        # 生成器总损失（对抗损失 + 内容损失）
        g_loss = g_loss_adv + g_loss_content
        g_loss.backward()
        self.gen_optimizer.step()

        return d_loss.item(), g_loss.item(), g_loss_content.item()

    def generate(self, x):
        self.generator.eval()
        with torch.no_grad():
            return self.generator(x.to(device))


# 可视化函数（与原始代码相同）
def plot_comparison(y_true, y_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='True Values')
    plt.plot(y_pred, label='Predicted Values')
    plt.title('GAN Prediction Comparison')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    plt.close()


# 模型评估函数（与原始代码相同）
def evaluate_model(model, test_loader):
    y_true = []
    y_pred = []

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        pred = model.generate(x)
        y_true.append(y.cpu().numpy())
        y_pred.append(pred.squeeze().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    print('\nGAN Evaluation Results:')
    print(f'R² Score: {r2_score(y_true, y_pred):.4f}')
    print(f'RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}')
    print(f'MAE: {mean_absolute_error(y_true, y_pred):.4f}')

    plot_comparison(y_true, y_pred)


# 训练GAN模型
def train_gan(gan, train_loader, epochs, save_interval=10):
    d_losses = []
    g_losses = []
    content_losses = []

    for epoch in range(epochs):
        d_epoch_loss = 0.0
        g_epoch_loss = 0.0
        content_epoch_loss = 0.0
        batches = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            d_loss, g_loss, content_loss = gan.train_step(x, y)
            d_epoch_loss += d_loss
            g_epoch_loss += g_loss
            content_epoch_loss += content_loss
            batches += 1

        d_epoch_loss /= batches
        g_epoch_loss /= batches
        content_epoch_loss /= batches

        d_losses.append(d_epoch_loss)
        g_losses.append(g_epoch_loss)
        content_losses.append(content_epoch_loss)

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"D Loss: {d_epoch_loss:.4f} | "
              f"G Loss: {g_epoch_loss:.4f} | "
              f"Content Loss: {content_epoch_loss:.4f}")

        # 定期保存模型
        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            torch.save(gan.generator.state_dict(), f'generator_epoch_{epoch + 1}.pth')
            torch.save(gan.discriminator.state_dict(), f'discriminator_epoch_{epoch + 1}.pth')

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(content_losses, label='Content Loss')
    plt.title('GAN Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('gan_training_losses.png')
    plt.close()

    return gan


# 主程序
if __name__ == "__main__":
    # 数据加载与预处理
    pp = pd.read_csv(r"C:\Python\DS\gt_2015.csv").values

    # 数据划分（与原始代码相同）
    x = pp[:, 1:-2].astype(np.float32)
    y = pp[:, -1].astype(np.float32)

    # 划分数据集
    x_train = x[:6000, :]
    y_train = y[:6000]
    x_test = x[6000:7500, :]
    y_test = y[6000:7500]

    # 验证集划分（改为全部用于训练）
    x_val, _, y_val, _ = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42
    )

    # 数据标准化（与原始代码相同）
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    # 转换为三维张量 [samples, seq_len, features]（与原始代码相同）
    x_train = x_train.reshape(-1, x_train.shape[1], 1)
    x_val = x_val.reshape(-1, x_val.shape[1], 1)
    x_test = x_test.reshape(-1, x_test.shape[1], 1)

    # 创建数据加载器（与原始代码相同）
    train_dataset = Mydataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    val_dataset = Mydataset(torch.FloatTensor(x_val), torch.FloatTensor(y_val))
    test_dataset = Mydataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # 初始化生成器和判别器
    input_dim = 1
    hidden_dim = 128
    num_layers = 3
    output_dim = 1

    generator = Generator(input_dim, hidden_dim, num_layers, output_dim)
    discriminator = Discriminator(input_dim, hidden_dim, num_layers, output_dim)

    # 设置优化器
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    # 创建GAN模型
    gan = GANModel(generator, discriminator, g_optimizer, d_optimizer)

    # 训练GAN
    print("Starting GAN Training...")
    gan = train_gan(gan, train_loader, epochs=100, save_interval=20)

    # 保存最终模型
    torch.save(gan.generator.state_dict(), 'final_generator.pth')
    torch.save(gan.discriminator.state_dict(), 'final_discriminator.pth')

    # 评估生成器
    print("\nEvaluating GAN Generator...")
    evaluate_model(gan, test_loader)