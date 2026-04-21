"""LSTM Enhanced — PyTorch 2 层 LSTM，序列输入 expanding-window 预测。

输入：过去 SEQ_LEN=12 个月的 (CPI 环比增长率 + 宏观变量 + NLP 叙事特征)。
目标：当月 CPI 环比增长率。
每个 expanding-window 步重新训练（小网络 + 有限 epoch + 训练集 StandardScaler）。
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler

from utils.data_utils import (
    setup_plot_style, load_enhanced_data, evaluate_and_plot,
    save_predictions, save_metrics_to_csv, ENHANCED_OUTPUT_DIR,
)

MODEL_NAME = 'lstm_enhanced'

SEQ_LEN = 12
HIDDEN = 64
N_LAYERS = 2
LR = 1e-3
EPOCHS = 60
BATCH = 32
RETRAIN_EVERY = 6  # 每 6 步重新训练一次，其余步用上次权重前向（加速）
SEED = 42


class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden=HIDDEN, num_layers=N_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def _build_sequences(feature_df, target_sr, seq_len):
    """返回 X: (N, seq_len, F)，y: (N,)，idx: 目标时间戳。"""
    idx = target_sr.index
    X, y, idxs = [], [], []
    vals = feature_df.values
    tgt = target_sr.values
    for t in range(seq_len, len(target_sr)):
        X.append(vals[t - seq_len:t])
        y.append(tgt[t])
        idxs.append(idx[t])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), pd.DatetimeIndex(idxs)


def _train(model, X_tr, y_tr, device):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    n = len(X_tr)
    X_t = torch.from_numpy(X_tr).to(device)
    y_t = torch.from_numpy(y_tr).to(device)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH):
            ids = perm[i:i+BATCH]
            opt.zero_grad()
            pred = model(X_t[ids])
            loss = loss_fn(pred, y_t[ids])
            loss.backward()
            opt.step()


def run_lstm_enhanced(file_path=None, start_ratio=0.8):
    setup_plot_style()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df, target_col = load_enhanced_data(file_path, stationarize=True)
    feature_df = df.drop(columns=[target_col])
    target_sr = df[target_col]

    # 以特征 + 目标一起作为序列输入（把当月 target 也作为特征之一，但目标本身来自单列）。
    # 为避免泄漏：输入序列是 t-SEQ_LEN .. t-1，目标是 t。因此把 target lag1 一起纳入特征。
    full_feat = feature_df.copy()
    full_feat[f'{target_col}_lag0in_seq'] = target_sr  # 序列窗口是 t-SEQ_LEN..t-1，这列在窗口内合法

    split_index = int(len(target_sr) * start_ratio)
    if split_index <= SEQ_LEN:
        raise ValueError("训练集不足以构造序列。")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f">>> LSTM device: {device}")
    print(f">>> Expanding-window (start_ratio={start_ratio}, seq_len={SEQ_LEN}, retrain_every={RETRAIN_EVERY})")

    n_features = full_feat.shape[1]
    model = LSTMRegressor(n_features=n_features).to(device)
    scaler = None

    preds, truths, idxs = [], [], []
    last_train_at = -10**9

    for t in range(split_index, len(target_sr)):
        # 训练集：0..t
        if (t - last_train_at) >= RETRAIN_EVERY:
            scaler = StandardScaler()
            feat_scaled = scaler.fit_transform(full_feat.iloc[:t].values)
            feat_scaled_df = pd.DataFrame(feat_scaled, index=full_feat.index[:t], columns=full_feat.columns)
            X_tr, y_tr, _ = _build_sequences(feat_scaled_df, target_sr.iloc[:t], SEQ_LEN)
            if len(X_tr) < BATCH:
                continue
            model = LSTMRegressor(n_features=n_features).to(device)
            _train(model, X_tr, y_tr, device)
            last_train_at = t

        # 预测第 t 步：窗口是 full_feat[t-SEQ_LEN:t]
        window_raw = full_feat.iloc[t - SEQ_LEN:t].values
        window_scaled = scaler.transform(window_raw).astype(np.float32)[None, :, :]
        model.eval()
        with torch.no_grad():
            yhat = float(model(torch.from_numpy(window_scaled).to(device)).cpu().numpy().ravel()[0])
        preds.append(yhat)
        truths.append(float(target_sr.iloc[t]))
        idxs.append(target_sr.index[t])

    y_true = pd.Series(truths, index=pd.DatetimeIndex(idxs), name=target_col)
    y_pred = pd.Series(preds, index=pd.DatetimeIndex(idxs), name='y_pred')

    metrics = evaluate_and_plot(y_true, y_pred, 'LSTM (Enhanced)',
                                os.path.join(ENHANCED_OUTPUT_DIR, '7_lstm_enhanced.png'))
    save_predictions(MODEL_NAME, y_true, y_pred)
    save_metrics_to_csv(MODEL_NAME, **metrics)
    return metrics


if __name__ == "__main__":
    run_lstm_enhanced()
