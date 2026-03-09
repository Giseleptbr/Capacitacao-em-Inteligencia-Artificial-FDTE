# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from statsmodels.tsa.stattools import pacf

#gera a seguencia
def create_sequences(df_values, past_steps=48, horizon=1, features=None):
    n = len(df_values)
    if features is None:
        n_features = df_values.shape[1]
    else:
        n_features = len(features)
    X, y = [], []
    for i in range(past_steps, n - horizon + 1):
        X.append(df_values[i - past_steps:i, :])
        y.append(df_values[i:i + horizon, 0])  # assume target in column 0
    X = np.array(X)
    y = np.array(y)
    if horizon == 1:
        y = y.reshape(-1)
    return X, y

#Separa entre treinamento e teste
def train_val_test_split_time(X, y, train_frac=0.7, val_frac=0.15):
    n = X.shape[0]
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    return X_train, y_train, X_val, y_val, X_test, y_test

#métricas de desempenho
def evaluate_preds(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE(%)': mape}


# ========== LSTNA simples ==========
def build_lstm_baseline(seq_len, n_feats, units=64, dropout=0.2):
    inp = layers.Input(shape=(seq_len, n_feats))
    x = layers.LSTM(units, return_sequences=False)(inp)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation='linear')(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=optimizers.Adam(1e-3), loss='mse')
    return model


def escolher_lag_pacf(y, max_lag=40, alpha=0.05):
    """
    Retorna o número de lags significativos baseado na PACF.
    O lag retornado é aquele cujo valor está fora dos intervalos de confiança.
    """
    pacf_vals, conf = pacf(y, nlags=max_lag, alpha=alpha)

    lags_significados = []
    for lag in range(1, len(pacf_vals)):
        low, high = conf[lag]
        if pacf_vals[lag] < low or pacf_vals[lag] > high:
            lags_significados.append(lag)

    if len(lags_significados) == 0:
        return 1  # fallback seguro

    return max(lags_significados)

def persistence_forecast(X):
    # prediz o ultimo valor
    return X[:, -1, 0]

# ========== Mistura de especialistas lstm ==========
def build_mixture_of_experts(seq_len, n_feats, n_experts=3, expert_units=32, gating_units=32, dropout=0.2):
    """
    Returns a Keras Model that outputs a weighted sum of expert outputs.
    Each expert produces a scalar; gating network outputs softmax weights.
    """
    inp = layers.Input(shape=(seq_len, n_feats), name='input_seq')

    # Experts (share shape but separate weights)
    expert_outputs = []
    for k in range(n_experts):
        x = layers.LSTM(expert_units, return_sequences=False, name=f'expert_lstm_{k}')(inp)
        x = layers.Dropout(dropout, name=f'expert_dropout_{k}')(x)
        out_k = layers.Dense(1, activation='linear', name=f'expert_out_{k}')(x)
        expert_outputs.append(out_k)  # shape (batch, 1)

    experts_concat = layers.Concatenate(axis=1, name='experts_concat')(expert_outputs)  # (batch, n_experts)

    # Gating network: can use a separate small LSTM or a dense on pooled features
    # Option A: use last time step pooled features (e.g., mean)
    pooled = layers.GlobalAveragePooling1D(name='global_avg')(inp)
    g = layers.Dense(gating_units, activation='relu', name='g_dense1')(pooled)
    g = layers.Dropout(0.1, name='g_dropout')(g)
    gating_logits = layers.Dense(n_experts, activation=None, name='g_logits')(g)
    gating_soft = layers.Activation('softmax', name='gating_softmax')(gating_logits)  # (batch, n_experts)

    # Weighted sum: multiply gating weights with expert predictions
    # experts_concat is (batch, n_experts); gating_soft (batch, n_experts)
    weighted = layers.Multiply(name='gating_multiply')([experts_concat, gating_soft])
    final_out = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1, keepdims=True), name='gating_sum')(weighted)

    model = models.Model(inputs=inp, outputs=final_out)
    model.compile(optimizer=optimizers.Adam(1e-3), loss='mse')
    return model


#===============================================
# reshape para 2D : (n_windows * past_steps, n_feats)
X_train_flat = X_train.reshape(-1, n_feats)
scaler.fit(X_train_flat)
def scale_X(X):
    n_w, seq_len, nf = X.shape
    Xf = X.reshape(-1, nf)
    Xf = scaler.transform(Xf)
    return Xf.reshape(n_w, seq_len, nf)

X_train = scale_X(X_train)
X_val = scale_X(X_val)
X_test = scale_X(X_test)

#Cria a sequencia
past_steps = 48
horizon = 1
X_all, y_all = create_sequences(arr, past_steps=past_steps, horizon=horizon)

# Separa treinamento e teste
X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_time(X_all, y_all, 0.7, 0.15)

#==========================================================================
#Chama o LSTM
#=========================================================================
lstm = build_lstm_baseline(past_steps, n_feats, units=64)
lstm.summary()
es = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
lstm.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=80, batch_size=64, callbacks=[es], verbose=1)
y_pred_lstm = lstm.predict(X_test).reshape(-1)
print("LSTM metrics:", evaluate_preds(y_test, y_pred_lstm))

#=========================================================================
#Chama mistura de especialista
#===========================================================================
moe = build_mixture_of_experts(past_steps, n_feats, n_experts=3, expert_units=48, gating_units=32)
moe.summary()

# Train MOE
es2 = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
moe.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=120, batch_size=64, callbacks=[es2], verbose=1)

# Predict & evaluate
y_pred_moe = moe.predict(X_test).reshape(-1)
print("MOE metrics:", evaluate_preds(y_test, y_pred_moe))


#===================================================================================================
#Executa o baseline
#===================================================================================================
y_pred_base = persistence_forecast(X_test)
print("Baseline metrics:", evaluate_preds(y_test, y_pred_base))

#==============================================================================================
#===========================Inspeciona a saida da gating======================================
get_gating = models.Model(inputs=moe.input, outputs=moe.get_layer('gating_softmax').output)
gating_vals = get_gating.predict(X_test)  # (n_test, n_experts)
print("Gating mean weights per expert:", gating_vals.mean(axis=0))


#==================================================================================================
# Grafico distribuição da gating para um periodo
plt.figure(figsize=(10,4))
plt.plot(gating_vals[:200])
plt.title('Gating weights (first 200 test windows)')
plt.xlabel('window idx')
plt.ylabel('weight')
plt.legend([f'expert_{i}' for i in range(gating_vals.shape[1])])
plt.show()

# ========== Salvando os modelos ==========
moe.save('moe_lstm_model.h5')
lstm.save('lstm_baseline.h5')
import joblib
joblib.dump(scaler, 'scaler.save')
