import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import math

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------------- 模型核心组件 --------------------
class AutoCorrelation(nn.Module):
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        head, channel, length = values.shape[1], values.shape[2], values.shape[3]
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(values, -int(index[i]), -1)
            delays_agg += pattern * (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        return V.contiguous(), None


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        super(AutoCorrelationLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, L, H, -1)
        values = self.value_projection(values).view(B, L, H, -1)
        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        return self.out_projection(out.view(B, L, -1)), attn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        moving_mean = self.moving_avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        res = x - moving_mean
        return res, moving_mean


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, moving_avg, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        new_x, _ = self.attention(x, x, x, None)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = self.dropout(self.activation(self.conv1(x.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        res, _ = self.decomp2(x + y)
        return res


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff, moving_avg, dropout=0.1,
                 activation="relu"):
        super(DecoderLayer, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, padding=1,
                                    padding_mode='circular')
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross):
        x = x + self.dropout(self.self_attention(x, x, x, None)[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, None)[0])
        x, trend2 = self.decomp2(x)
        y = self.dropout(self.activation(self.conv1(x.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        x, trend3 = self.decomp3(x + y)
        residual_trend = trend1 + trend2 + trend3
        return x, self.projection(residual_trend.transpose(1, 2)).transpose(1, 2)


# -------------------- Autoformer模型 --------------------
class Autoformer(nn.Module):
    def __init__(self, config):
        super(Autoformer, self).__init__()
        self.pred_len = config.pred_len
        self.decomp = series_decomp(config.moving_avg)

        # 编码器
        self.enc_embedding = DataEmbedding(config.enc_in, config.d_model)
        self.encoder = nn.ModuleList([
            EncoderLayer(
                AutoCorrelationLayer(AutoCorrelation(), config.d_model, config.n_heads),
                config.d_model,
                config.d_ff,
                config.moving_avg,
                config.dropout,
                config.activation
            ) for _ in range(config.e_layers)
        ])

        # 解码器
        self.dec_embedding = DataEmbedding(config.dec_in, config.d_model)
        self.decoder = nn.ModuleList([
            DecoderLayer(
                AutoCorrelationLayer(AutoCorrelation(True), config.d_model, config.n_heads),
                AutoCorrelationLayer(AutoCorrelation(), config.d_model, config.n_heads),
                config.d_model,
                config.c_out,
                config.d_ff,
                config.moving_avg,
                config.dropout,
                config.activation
            ) for _ in range(config.d_layers)
        ])

        self.projection = nn.Linear(config.d_model, config.c_out)

    def forward(self, x):
        # 输入形状: [batch_size, seq_len, 1]
        seasonal_init, trend_init = self.decomp(x)
        enc_out = self.enc_embedding(x)

        for layer in self.encoder:
            enc_out = layer(enc_out)

        dec_out = self.dec_embedding(seasonal_init)
        trend = torch.zeros_like(dec_out)
        for layer in self.decoder:
            dec_out, residual_trend = layer(dec_out, enc_out)
            trend += residual_trend

        dec_out = self.projection(dec_out + trend)
        return dec_out[:, -self.pred_len:, :]


# -------------------- 配置类 --------------------
class Config:
    def __init__(self, feature_dim):
        self.enc_in = 1  # 输入特征维度
        self.dec_in = 1  # 解码器输入维度
        self.c_out = 1  # 输出维度
        self.seq_len = feature_dim  # 序列长度（特征数量）
        self.pred_len = 1  # 预测长度
        self.d_model = 32  # 模型维度
        self.n_heads = 2  # 注意力头数
        self.e_layers = 2  # 编码器层数
        self.d_layers = 1  # 解码器层数
        self.d_ff = 256  # 前馈网络维度
        self.moving_avg = 25 # 移动平均窗口
        self.dropout = 0.1  # Dropout概率
        self.activation = 'gelu'  # 激活函数


# -------------------- 数据管道 --------------------
class TSRegressionDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data).unsqueeze(-1)  # [样本数, 特征数, 1]
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# -------------------- 训练和评估函数 --------------------
def train_model(model, train_loader, val_loader, epochs=200, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x).squeeze()
            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x).squeeze()
                val_loss += criterion(output, y).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), 'best_autoformer.pth')

        print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}')


def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x).squeeze().cpu().numpy().flatten()
            y_true.extend(y.numpy().flatten())
            y_pred.extend(pred)

    print("\nEvaluation Results:")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:], label='True')
    plt.plot(y_pred[:], label='Predicted')
    plt.title("Autoformer Prediction Comparison")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    plt.close()



# -------------------- 主程序 --------------------
if __name__ == "__main__":
    # 加载数据
    data = pd.read_csv(r"C:\Python\DS\gt_2013.csv").values
    x = data[:, 1:-2].astype(np.float32)
    y = data[:, -1].astype(np.float32)

    # 数据划分
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
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # 创建配置
    feature_dim = x_train.shape[1]
    config = Config(feature_dim)

    # 创建数据集
    train_dataset = TSRegressionDataset(x_train, y_train)
    val_dataset = TSRegressionDataset(x_val, y_val)
    test_dataset = TSRegressionDataset(x_test, y_test)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # 初始化模型
    model = Autoformer(config)

    # 训练
    print("Starting Training...")
    train_model(model, train_loader, val_loader)

    # 评估
    print("\nEvaluating Best Model...")
    model.load_state_dict(
        torch.load('best_autoformer.pth',
                   map_location=device,
                   weights_only=True)  # 显式启用安全模式
    )
    evaluate_model(model, test_loader)