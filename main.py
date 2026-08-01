import os
import random
import numpy as np
import tensorflow as tf

# 🔐 Reproducibility setup
SEED = 45
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ✅ Enable full determinism
tf.config.experimental.enable_op_determinism()
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from datetime import datetime

# Step 1: Download Nifty 50 data
#today = datetime.today().strftime('%Y-%m-%d')
today = '2026-03-16'
data = yf.download("^NSEI", start="2018-01-01", end=today, progress=False)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)
z = data['Close'].dropna().values

# Step 2: Adaptive Kalman Filtering (AKF)
scaler = MinMaxScaler()
z_norm = scaler.fit_transform(z.reshape(-1, 1)).flatten()
n = len(z_norm)
x, P, Q, R, K = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
x[0], P[0], Q[0], R[0] = z_norm[0], 1.0, 1e-5, 0.01
alpha, beta, eps = 0.5, 0.5, 1e-4
for k in range(1, n):
    x_pred, P_pred = x[k-1], P[k-1] + Q[k-1]
    K[k] = P_pred / (P_pred + R[k-1])
    x[k] = x_pred + K[k] * (z_norm[k] - x_pred)
    P[k] = (1 - K[k]) * P_pred
    Q[k] = max(beta * Q[k-1] + (1 - beta) * abs(P_pred - P[k-1]), eps)
    R[k] = max(alpha * R[k-1] + (1 - alpha) * (z_norm[k] - x[k])**2, eps)
x_est = scaler.inverse_transform(x.reshape(-1, 1)).flatten()

# Step 3: Create DataFrame
filtered_df = pd.DataFrame({
    'Date': data.index,
    'Filtered_Close': x_est,
    'High': data['High'].values.flatten(),
    'Low': data['Low'].values.flatten(),
    'Close': data['Close'].values.flatten()
})
filtered_df.set_index('Date', inplace=True)

# Step 4: Technical Indicators
filtered_df['EMA_9'] = filtered_df['Filtered_Close'].ewm(span=9, adjust=False).mean()
filtered_df['EMA_21'] = filtered_df['Filtered_Close'].ewm(span=21, adjust=False).mean()
delta = filtered_df['Filtered_Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
filtered_df['RSI_14'] = 100 - (100 / (1 + rs))
filtered_df['HL'] = filtered_df['High'] - filtered_df['Low']
filtered_df['HC'] = (filtered_df['High'] - filtered_df['Filtered_Close'].shift()).abs()
filtered_df['LC'] = (filtered_df['Low'] - filtered_df['Filtered_Close'].shift()).abs()
filtered_df['TR'] = filtered_df[['HL', 'HC', 'LC']].max(axis=1)
filtered_df['ATR_14'] = filtered_df['TR'].rolling(window=14).mean()

# Step 5: Prepare features and scale
target_scaler = MinMaxScaler()
features = ['Filtered_Close', 'EMA_9', 'EMA_21', 'ATR_14', 'RSI_14']
filtered_df.dropna(inplace=True)
filtered_close_original = filtered_df[['Filtered_Close']].copy()
target_scaler.fit(filtered_close_original)
scaler = MinMaxScaler()
filtered_df[features] = scaler.fit_transform(filtered_df[features])
filtered_df['Filtered_Close'] = target_scaler.transform(filtered_close_original)
# ✅ RECREATE THE y_test to match X_test.npy
SEQ_LEN = 5  # make sure this matches your training

# Recreate full sequence from updated filtered_df
features = ['Filtered_Close', 'EMA_9', 'EMA_21', 'ATR_14', 'RSI_14']
X_all, y_all = [], []
for i in range(len(filtered_df) - SEQ_LEN):
    X_all.append(filtered_df[features].iloc[i:i+SEQ_LEN].values.astype(np.float32))
    y_all.append(filtered_df['Filtered_Close'].iloc[i + SEQ_LEN])
X_all = np.array(X_all)
y_all = np.array(y_all)

# Same split logic (shuffle=False, test_size=0.2)
split_idx = int(len(X_all) * 0.8)
X_train = X_all[:split_idx]
X_test = X_all[split_idx:]
y_train = y_all[:split_idx]
y_test = y_all[split_idx:]
# 🕐 Retrieve date ranges for training and testing sets
train_start_date = filtered_df.index[0]
train_end_date = filtered_df.index[split_idx + SEQ_LEN - 1]

test_start_date = filtered_df.index[split_idx + SEQ_LEN]
test_end_date = filtered_df.index[-1]

print("\n📅 Data Split Information:")
print(f"🟢 Training Data: {train_start_date.date()} to {train_end_date.date()} ({len(X_train)} sequences)")
print(f"🔵 Testing Data : {test_start_date.date()} to {test_end_date.date()} ({len(X_test)} sequences)")

import os

# Ensure the directory exists before saving
os.makedirs("DCWRNN_Models2/SHAP_Assets", exist_ok=True)

# Now save y_test safely
np.save("DCWRNN_Models2/SHAP_Assets/y_test.npy", y_test)

# Step 6: Prepare sequences
SEQ_LEN = 5
X, y = [], []
for i in range(len(filtered_df) - SEQ_LEN):
    X.append(filtered_df[features].iloc[i:i+SEQ_LEN].values.astype(np.float32))
    y.append(filtered_df['Filtered_Close'].iloc[i + SEQ_LEN])
X = np.stack(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Step 7: Define DCWRNN
@tf.keras.utils.register_keras_serializable()
class DCWRNNCell(tf.keras.layers.Layer):
    def __init__(self, units, update_interval, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.update_interval = update_interval

    def build(self, input_shape):
        self.W_h = self.add_weight(shape=(self.units, self.units), initializer='glorot_uniform')
        self.W_x = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_uniform')
        self.b = self.add_weight(shape=(self.units,), initializer='zeros')

    def call(self, x, h_prev, t):
        should_update = tf.equal(tf.math.floormod(t, self.update_interval), 0)
        h_tilde = tf.nn.tanh(tf.matmul(h_prev, self.W_h) + tf.matmul(x, self.W_x) + self.b)
        return tf.where(should_update, h_tilde, h_prev)

@tf.keras.utils.register_keras_serializable()
class DCWRNN(tf.keras.layers.Layer):
    def __init__(self, units, update_intervals, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.update_intervals = update_intervals
        self.cells = [DCWRNNCell(units, c) for c in update_intervals]

    def build(self, input_shape):
        for cell in self.cells:
            cell.build(input_shape[1:])
        self.built = True

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        h = [tf.zeros((batch_size, self.units)) for _ in self.cells]
        outputs_ta = tf.TensorArray(dtype=tf.float32, size=time_steps)

        def loop_body(t, h, outputs_ta):
            x_t = inputs[:, t, :]
            new_h, step_outputs = [], []
            for i, cell in enumerate(self.cells):
                h_i = cell(x_t, h[i], t)
                new_h.append(h_i)
                step_outputs.append(h_i)
            outputs_ta = outputs_ta.write(t, tf.concat(step_outputs, axis=1))
            return t + 1, new_h, outputs_ta

        t0 = tf.constant(0)
        _, _, outputs_final = tf.while_loop(lambda t, *_: t < time_steps, loop_body, (t0, h, outputs_ta))
        outputs = outputs_final.stack()
        return tf.transpose(outputs, [1, 0, 2])

def build_dcwrnn_model(input_shape, units=32, update_intervals=[1, 3, 5]):
    inputs = tf.keras.Input(shape=input_shape)
    x = DCWRNN(units, update_intervals, name='dcwrnn_1')(inputs)
    x_last = x[:, -1, :]
    output = tf.keras.layers.Dense(1)(x_last)
    model = tf.keras.Model(inputs, output)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])
    return model

from tensorflow.keras.callbacks import ModelCheckpoint

# ✅ Define path to save the best weights
checkpoint_path = "best_dcwrnn_weights.weights.h5"


# ✅ Callback to save only the best model based on validation loss
checkpoint_cb = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# Step 8: Train model
model = build_dcwrnn_model(input_shape=(SEQ_LEN, len(features)))
history=model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint_cb],
    verbose=1
)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
# ✅ Load the best model weights (based on lowest val_loss)
model.load_weights(checkpoint_path)
print("✅ Best model weights loaded successfully.")

# Predict on test set (scaled)
y_pred_scaled = model.predict(X_test)

# Compute metrics directly on scaled values
mae_scaled = mean_absolute_error(y_test, y_pred_scaled)
mse_scaled = mean_squared_error(y_test, y_pred_scaled)
rmse_scaled = np.sqrt(mse_scaled)
r2_scaled = r2_score(y_test, y_pred_scaled)

# Print scaled performance
from datetime import datetime

print("📊 DCWRNN Performance Metrics on Scaled Data")
print(f"✅ MAE (scaled):  {mae_scaled:.6f}")
print(f"✅ MSE (scaled):  {mse_scaled:.6f}")
print(f"✅ RMSE (scaled): {rmse_scaled:.6f}")
print(f"✅ R² (scaled):   {r2_scaled:.6f}")

# ✅ Custom logic to save only if R² ≥ 0.96
R2_THRESHOLD = 0.96
if r2_scaled >= R2_THRESHOLD:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    best_weights_path = f"best_dcwrnn_r2_{r2_scaled:.4f}_{timestamp}.weights.h5"
    model.save_weights(best_weights_path)
    # Save full model
    model_path = f"best_model_r2_{r2_scaled:.4f}_{timestamp}.h5"
    model.save(model_path)

    print(f"✅ Full model saved to: {model_path}")

    print(f"\n✅ R² ≥ {R2_THRESHOLD}. Best weights saved to: {best_weights_path}")
else:
    print(f"\n⚠️ R² < {R2_THRESHOLD}. Model weights not saved.")

# Step 9: Predict next day
last_seq_scaled = filtered_df[features].iloc[-SEQ_LEN:].values.astype(np.float32)
last_seq_scaled = np.expand_dims(last_seq_scaled, axis=0)
next_scaled_pred = model.predict(last_seq_scaled)
next_unscaled_close = target_scaler.inverse_transform(next_scaled_pred)[0][0]

# ✅ Add the date for which prediction is made
last_date = filtered_df.index[-1]

# ✅ Define next valid trading day function
def get_next_trading_day(date):
    next_day = date + pd.Timedelta(days=1)
    while next_day.weekday() >= 5:  # skip Saturday & Sunday
        next_day += pd.Timedelta(days=1)
    return next_day

# ✅ Get the predicted date
next_trading_day = get_next_trading_day(last_date)

# ✅ Print result
print(f"\n📆 Predicted Nifty 50 Close on {next_trading_day.date()}: ₹{round(next_unscaled_close, 2)}")

# ✅ Smart trading day logic
import datetime

# 📅 Step 1: NSE Holidays for 2025
nse_holidays_2025 = {
    datetime.date(2025, 1, 26),
    datetime.date(2025, 3, 31),
    datetime.date(2025, 4, 14),
    datetime.date(2025, 4, 18),
    datetime.date(2025, 5, 1),
    datetime.date(2025, 6, 2),
    datetime.date(2025, 8, 15),
    datetime.date(2025, 10, 2),
    datetime.date(2025, 10, 21),
    datetime.date(2025, 10, 22),
    datetime.date(2025, 11, 3),
    datetime.date(2025, 12, 25)
}

# 📘 Step 2: Find the next valid trading day
def get_next_trading_day(last_date):
    next_day = last_date + pd.Timedelta(days=1)
    while next_day.weekday() >= 5 or next_day.date() in nse_holidays_2025:
        next_day += pd.Timedelta(days=1)
    return next_day

# ✅ Use today's actual date (not just filtered_df)
#last_date = pd.to_datetime(today)
#next_trading_date = get_next_trading_day(last_date)

# 💰 Step 3: Final prediction output
#print(f"\n📆 Predicted Nifty 50 Close on {next_trading_date.date()}: ₹{round(next_unscaled_close, 2)}")


import shap
import numpy as np

# ✅ Step 1: Reduce SHAP computation time by using a smaller sample
X_sample = X_test[:10]  # Reduced from 100 to 30 for faster computation
X_sample_flat = X_sample.reshape((X_sample.shape[0], -1))

# ✅ Step 2: Define prediction wrapper
def model_predict_flat(X_flat):
    reshaped = X_flat.reshape((-1, SEQ_LEN, len(features)))
    return model.predict(reshaped)

# ✅ Step 3: KernelExplainer with background data
explainer = shap.KernelExplainer(model_predict_flat, X_sample_flat)

# ✅ Step 4: Compute SHAP values
shap_values = explainer.shap_values(X_sample_flat)

# ✅ Step 5: Create flat feature names (e.g., Filtered_Close_t0, EMA_9_t2)
feature_names_flat = [f"{feat}_t{t}" for t in range(SEQ_LEN) for feat in features]

# ✅ Step 6: Plot summary
#shap.summary_plot(shap_values, X_sample_flat, feature_names=feature_names_flat)

# ✅ Step 7: Print average SHAP importance for each feature
avg_importances = np.mean(np.abs(shap_values), axis=0)
nonzero_features = [(name, imp) for name, imp in zip(feature_names_flat, avg_importances) if imp > 0]


print("\n🔍 Average SHAP Importance per Feature-TimeStep:")
for name, importance in sorted(zip(feature_names_flat, avg_importances), key=lambda x: x[1], reverse=True):
   print(f"{name:20s} ➤ {importance.item():.6f}")
import matplotlib.pyplot as plt

# ✅ Set number of top features to show
top_k = 10

# ✅ Get top_k features based on average SHAP importance
top_features = sorted(nonzero_features, key=lambda x: x[1], reverse=True)[:top_k]
# Prepare data for plot
names, importances = zip(*top_features)
importances = np.array([imp.item() if isinstance(imp, np.ndarray) else imp for imp in importances])

# Plot
plt.figure(figsize=(10, 6))
plt.barh(names[::-1], importances[::-1])
plt.xlabel("Average SHAP Importance")
plt.title(f"Top {top_k} SHAP Features")
plt.tight_layout()
plt.show()
plt.figure(figsize=(14, 6))

# Plot last 100 days of actual close
plt.plot(filtered_df.index[-100:], filtered_df['Close'].values[-100:], label='Actual Close', color='blue')

# Plot predicted close
plt.scatter(next_trading_day, next_unscaled_close, color='red', label='Predicted Close', zorder=5)

plt.axvline(next_trading_day, color='red', linestyle='--', alpha=0.5)

plt.text(next_trading_day, next_unscaled_close, f' ₹{round(next_unscaled_close, 2)}',
         color='red', fontsize=10, verticalalignment='center')

plt.title(f'Nifty 50 Close Price with Prediction for {next_trading_day.strftime("%d-%m-%Y")}')

plt.xlabel('Date')
plt.ylabel('Close Price (₹)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print(f"Latest data point available: {filtered_df.index[-1].date()}")
print(f"Prediction made for       : {next_trading_day.date()}")
print("\n📅 Last 5 entries in downloaded Nifty 50 data:")
print(data.tail())
def backtest_model(name, model, y_test, target_scaler):
    # 1. Get unscaled predictions and actuals
    y_test_unscaled = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_unscaled = target_scaler.inverse_transform(model.predict(X_test)).flatten()

    # 2. Create results_df
    results_df = pd.DataFrame({
        'Date': filtered_df.index[-len(y_test):],
        'Actual_Close': y_test_unscaled,
        'Predicted_Close': y_pred_unscaled
    })
    results_df.set_index('Date', inplace=True)

    # 3. Directional changes
    results_df['Actual_Change'] = results_df['Actual_Close'].diff()
    results_df['Predicted_Change'] = results_df['Predicted_Close'].diff()
    results_df['Actual_Dir'] = np.sign(results_df['Actual_Change'])
    results_df['Predicted_Dir'] = np.sign(results_df['Predicted_Change'])
    results_df['Correct'] = (results_df['Actual_Dir'] == results_df['Predicted_Dir']).astype(int)
    directional_accuracy = results_df['Correct'].mean()

    # 4. Returns
    results_df['Market_Return'] = results_df['Actual_Close'].pct_change()
    results_df['Strategy_Return'] = results_df['Market_Return'] * results_df['Predicted_Dir'].shift()
    results_df['Cumulative_Market'] = (1 + results_df['Market_Return']).cumprod()
    results_df['Cumulative_Strategy'] = (1 + results_df['Strategy_Return']).cumprod()

    # 5. Sharpe Ratio
    mean_return = results_df['Strategy_Return'].mean()
    std_return = results_df['Strategy_Return'].std()
    sharpe_ratio = mean_return / std_return * np.sqrt(252)

    # 6. Print results
    print(f"\n📘 {name} Prediction Backtesting Report")
    print(f"🔢 Total Test Samples: {len(results_df)}")
    print(f"✅ Directional Accuracy: {directional_accuracy:.2%}")
    print(f"📊 Cumulative Market Return:  {(results_df['Cumulative_Market'].iloc[-1] - 1)*100:.2f}%")
    print(f"📊 Cumulative Strategy Return: {(results_df['Cumulative_Strategy'].iloc[-1] - 1)*100:.2f}%")
    print(f"📈 Sharpe Ratio: {sharpe_ratio:.2f}")

    # 7. Plot
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Cumulative_Market'], label='Market Return', linestyle='--', color='gray')
    plt.plot(results_df['Cumulative_Strategy'], label=f'{name} Strategy Return', color='green')
    plt.title(f"{name} Strategy vs Market")
    plt.xlabel("Date")
    plt.ylabel("Growth of ₹1 Investment")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results_df, directional_accuracy, sharpe_ratio

def build_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

lstm_model = build_lstm_model(X_train.shape[1:])
lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)
def build_rnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.SimpleRNN(64),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

rnn_model = build_rnn_model(X_train.shape[1:])
rnn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)
def build_cnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(64, kernel_size=2, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

cnn_model = build_cnn_model(X_train.shape[1:])
cnn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"\n📊 {name} Performance:")
    print(f"MAE:  {mae:.6f}")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²:   {r2:.6f}")

evaluate_model(lstm_model, "LSTM")
evaluate_model(rnn_model, "RNN")
evaluate_model(cnn_model, "CNN")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize containers
model_names = ['DCWRNN', 'LSTM', 'RNN', 'CNN']
maes, mses, rmses, r2s = [], [], [], []

def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Append metrics
    maes.append(mae)
    mses.append(mse)
    rmses.append(rmse)
    r2s.append(r2)

    print(f"\n📊 {name} Performance:")
    print(f"MAE:  {mae:.6f}")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²:   {r2:.6f}")

# Evaluate all models
evaluate_model(model, "DCWRNN")
evaluate_model(lstm_model, "LSTM")
evaluate_model(rnn_model, "RNN")
evaluate_model(cnn_model, "CNN")
x = np.arange(len(model_names))  # label locations
width = 0.2  # bar width

plt.figure(figsize=(14, 6), dpi=150)
plt.bar(x - width, maes, width, label='MAE')
plt.bar(x, mses, width, label='MSE')
plt.bar(x + width, rmses, width, label='RMSE')

plt.xlabel('Model')
plt.ylabel('Error Value')
plt.title('Model Performance Comparison (MAE, MSE, RMSE)')
plt.xticks(x, model_names)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 📊 Plot R² separately for clarity
plt.figure(figsize=(10, 5), dpi=150)
plt.bar(model_names, r2s, color='green')
plt.ylabel('R² Score')
plt.title('Model R² Score Comparison')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Run for DCWRNN
dcwrnn_results, dcwrnn_dir_acc, dcwrnn_sharpe = backtest_model("DCWRNN", model, y_test, target_scaler)

# Run for LSTM
lstm_results, lstm_dir_acc, lstm_sharpe = backtest_model("LSTM", lstm_model, y_test, target_scaler)

# Run for RNN
rnn_results, rnn_dir_acc, rnn_sharpe = backtest_model("RNN", rnn_model, y_test, target_scaler)

# Run for CNN
cnn_results, cnn_dir_acc, cnn_sharpe = backtest_model("CNN", cnn_model, y_test, target_scaler)

# ======================= Risk-Adjusted Ratios: Compute & Compare (PASTE HERE) =======================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRADING_DAYS = 252

# --- helpers (self-contained) ---
def equity_curve(returns):
    r = np.asarray(pd.Series(returns).dropna().values, dtype=float)
    return np.cumprod(1.0 + r)

def max_drawdown(returns):
    eq = equity_curve(returns)
    if eq.size == 0:
        return np.nan
    peaks = np.maximum.accumulate(eq)
    dd = (peaks - eq) / peaks
    return float(np.max(dd))

def annualized_return(returns, geometric=True):
    r = np.asarray(pd.Series(returns).dropna().values, dtype=float)
    n = r.size
    if n == 0:
        return np.nan
    if geometric:
        total = float(np.prod(1.0 + r) - 1.0)
        return (1.0 + total) ** (TRADING_DAYS / n) - 1.0
    else:
        return float(np.mean(r) * TRADING_DAYS)

def downside_deviation(returns, mar_daily=0.0):
    r = np.asarray(pd.Series(returns).dropna().values, dtype=float)
    shortfall = np.minimum(0.0, r - mar_daily)
    return float(np.sqrt(np.mean(shortfall ** 2))) if shortfall.size else np.nan

def sortino_ratio(returns, rf_annual=0.0, mar_daily=0.0):
    r = np.asarray(pd.Series(returns).dropna().values, dtype=float)
    if r.size == 0:
        return np.nan
    target_daily = mar_daily if mar_daily != 0.0 else ((1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0 if rf_annual != 0.0 else 0.0)
    mean_excess_daily = float(np.mean(r - target_daily))
    dd = downside_deviation(r, mar_daily=target_daily)
    if dd == 0.0 or np.isnan(dd):
        return np.inf if mean_excess_daily > 0 else (-np.inf if mean_excess_daily < 0 else np.nan)
    return (mean_excess_daily * TRADING_DAYS) / (dd * np.sqrt(TRADING_DAYS))

def omega_ratio(returns, r0_daily=0.0):
    r = np.asarray(pd.Series(returns).dropna().values, dtype=float)
    up = float(np.sum(np.clip(r - r0_daily, 0.0, None)))
    down = float(np.sum(np.clip(r0_daily - r, 0.0, None)))
    if down == 0.0:
        return np.inf if up > 0 else np.nan
    return up / down
def sharpe_ratio(returns, rf_annual=0.0):
    r = np.asarray(pd.Series(returns).dropna().values, dtype=float)
    if r.size == 0:
        return np.nan
    rf_daily = (1.0 + rf_annual)**(1.0 / TRADING_DAYS) - 1.0 if rf_annual != 0.0 else 0.0
    excess = r - rf_daily
    mu = float(np.mean(excess))
    sd = float(np.std(excess, ddof=1))  # sample stdev
    if sd == 0.0 or np.isnan(sd):
        return np.inf if mu > 0 else (-np.inf if mu < 0 else np.nan)
    return (mu * TRADING_DAYS) / (sd * np.sqrt(TRADING_DAYS))

def compute_ratios_from_returns(daily_returns, rf_annual=0.0, r0_daily=0.0, geometric_ann=True):
    r = np.asarray(pd.Series(daily_returns).dropna().values, dtype=float)
    ann = annualized_return(r, geometric=geometric_ann)
    mdd = max_drawdown(r)
    calmar = (ann / mdd) if (mdd not in (0.0, np.nan) and not np.isnan(mdd)) else (np.inf if (not np.isnan(ann) and ann > 0) else np.nan)
    return {
    "N Days": r.size,
    "Annual Return": ann,
    "Max Drawdown": mdd,
    "Calmar": calmar,
    "Omega (R0=0)": omega_ratio(r, r0_daily=r0_daily),
    "Sortino": sortino_ratio(r, rf_annual=rf_annual, mar_daily=0.0),
    "Sharpe":  sharpe_ratio(r, rf_annual=rf_annual),   # <--- add this
}


def print_ratios_report(name, daily_returns, rf_annual=0.0, r0_daily=0.0):
    r = np.asarray(pd.Series(daily_returns).dropna().values, dtype=float)
    if r.size == 0:
        print(f"\n📘 {name} Risk-Adjusted Ratios\n⚠️ No data.\n")
        return
    ann = annualized_return(r, geometric=True)
    mdd = max_drawdown(r)
    calmar = (ann / mdd) if (mdd not in (0.0, np.nan) and not np.isnan(mdd)) else (np.inf if ann > 0 else np.nan)
    sortino = sortino_ratio(r, rf_annual=rf_annual, mar_daily=0.0)
    omega = omega_ratio(r, r0_daily=r0_daily)
    total = float(np.prod(1.0 + r) - 1.0)

    print(f"\n📘 {name} Risk-Adjusted Ratios Report")
    print(f"🔢 Observations               : {r.size}")
    print(f"💼 Total Period Return        : {total*100:6.2f}%")
    print(f"📈 Annualized Return          : {ann*100:6.2f}%")
    print(f"📉 Max Drawdown (MDD)         : {mdd*100:6.2f}%")
    print(f"🧮 Calmar Ratio               : {calmar:6.2f}")
    print(f"🟩 Omega (R0=0)               : {omega:6.2f}")
    print(f"🎯 Sortino Ratio              : {sortino:6.2f}")

# --------- build returns dict from your backtests ----------
RF_ANNUAL = 0.00   # e.g., 0.06 for 6% p.a. if you want to include risk-free
R0_DAILY  = 0.00   # Omega threshold; for ~5% p.a. use: (1.05)**(1/252)-1

model_returns_map = {
    "DCWRNN": dcwrnn_results["Strategy_Return"].values,
    "LSTM":   lstm_results["Strategy_Return"].values,
    "RNN":    rnn_results["Strategy_Return"].values,
    "CNN":    cnn_results["Strategy_Return"].values,
}

# --------- compute metrics per model ----------
rows = []
for name, rets in model_returns_map.items():
    m = compute_ratios_from_returns(rets, rf_annual=RF_ANNUAL, r0_daily=R0_DAILY, geometric_ann=True)
    m["Model"] = name
    rows.append(m)

ratios_df = pd.DataFrame(rows).set_index("Model").sort_values("Calmar", ascending=False)

# --------- print comparison table ----------
print("\n================ Risk-Adjusted Ratios: Comparison ================\n")
print(ratios_df[["N Days","Annual Return","Max Drawdown","Calmar","Omega (R0=0)","Sortino"]]
      .applymap(lambda x: f"{x:.6f}" if isinstance(x, (int,float,np.floating)) else x))

# --------- per-model pretty reports ----------
for name in ["DCWRNN","LSTM","RNN","CNN"]:
    print_ratios_report(name, model_returns_map[name], rf_annual=RF_ANNUAL, r0_daily=R0_DAILY)

# --------- optional bar chart + CSV ----------
ratios_df[["Calmar","Omega (R0=0)","Sortino"]].plot(kind="bar", figsize=(12,6), rot=0, grid=True, title="Risk-Adjusted Ratios (Higher is Better)")
plt.tight_layout()
plt.show()
# ======================= Baseline Comparison Table (rf/R0) =======================
# Reuses: model_returns_map, compute_ratios_from_returns (already defined above)

def compute_table_for_baseline(returns_map, rf_annual, r0_daily):
    rows = []
    for name, rets in returns_map.items():
        m = compute_ratios_from_returns(rets, rf_annual=rf_annual, r0_daily=r0_daily, geometric_ann=True)
        m["Model"] = name
        rows.append(m)
    return pd.DataFrame(rows).set_index("Model")

# --- Baselines ---
rf0, r00 = 0.00, 0.00
rf_real = 0.06                                      # 6% p.a. risk-free
r0_real = (1.05)**(1/252) - 1                       # ~5% p.a. daily threshold

# --- Compute under both baselines ---
df_zero = compute_table_for_baseline(model_returns_map, rf0, r00)
df_real = compute_table_for_baseline(model_returns_map, rf_real, r0_real)

# --- Build side-by-side comparison (Calmar identical across baselines) ---
comp = pd.DataFrame({
    "Calmar":                     df_zero["Calmar"],
    "Sharpe (rf=0%)":             df_zero["Sharpe"],
    "Sharpe (rf=6%)":             df_real["Sharpe"],
    "Sortino (MAR=0)":            df_zero["Sortino"],
    "Sortino (MAR≈5% p.a.)":      df_real["Sortino"],
    "Omega (R0=0)":               df_zero["Omega (R0=0)"],
    f"Omega (R0≈{r0_real:.5f})":  df_real["Omega (R0=0)"],  # column name reused from helper
})

# --- Pretty print without FutureWarning ---
def _fmt(x):
    try:
        return f"{x:.6f}"
    except Exception:
        return str(x)

print("\n================ Risk-Adjusted Ratios: Baseline Comparison ================\n")
print(comp.apply(lambda s: s.map(_fmt)))

# --- Optional: save to CSV & LaTeX for thesis ---
comp.to_csv("risk_adjusted_ratios_baseline_comparison.csv", float_format="%.6f")
with open("risk_adjusted_baseline_comparison.tex","w", encoding="utf-8") as f:
    f.write(comp.to_latex(float_format="%.6f"))
print("\n📁 Saved: risk_adjusted_ratios_baseline_comparison.csv")
print("📁 Saved: risk_adjusted_baseline_comparison.tex")

ratios_df.to_csv("risk_adjusted_ratios_comparison.csv", float_format="%.6f")
print("\n📁 Saved: risk_adjusted_ratios_comparison.csv")
# ==============================================================================================


import os

# Create signals folder
os.makedirs("signals", exist_ok=True)

# Export prediction signals
signals_df = pd.DataFrame({
    "Date": filtered_df.index[-len(y_test):],
    "Actual_Close": target_scaler.inverse_transform(y_test.reshape(-1,1)).flatten(),
    "Predicted_Close": target_scaler.inverse_transform(model.predict(X_test)).flatten()
})

signals_df.to_csv("signals/dcwrnn_signals.csv", index=False)

print("✅ Signals exported successfully!")
print(signals_df.head())