# ============================================================
# PHASE 0
# DCWRNN NIFTY 50 PRICE PREDICTION
# Leakage-Controlled + Naive Baseline
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

SEED = 45

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


# ============================================================
# 3. CREATE FOLDERS
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("graphs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("signals", exist_ok=True)


# ============================================================
# 4. PARAMETERS
# ============================================================

TICKER = "^NSEI"

START_DATE = "2018-01-01"
END_DATE = "2026-03-16"

SEQ_LEN = 5

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.0005

DCWRNN_UNITS = 32
UPDATE_INTERVALS = [1, 3, 5]


# ============================================================
# 5. DOWNLOAD NIFTY 50 DATA
# ============================================================

print("\n" + "=" * 70)
print("DOWNLOADING NIFTY 50 DATA")
print("=" * 70)

data = yf.download(
    TICKER,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=False,
    progress=False
)

if data.empty:
    raise ValueError("No data was downloaded from Yahoo Finance.")


# ============================================================
# 6. HANDLE MULTIINDEX COLUMNS
# ============================================================

if isinstance(data.columns, pd.MultiIndex):

    data.columns = data.columns.get_level_values(0)

    # Remove duplicate columns if any
    data = data.loc[:, ~data.columns.duplicated()]


required_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


data = data[required_columns].copy()

data.index = pd.to_datetime(data.index)

data = data.sort_index()

data = data.dropna()


print("Downloaded rows:", len(data))
print("Start date:", data.index.min())
print("End date:", data.index.max())


# ============================================================
# 7. SAVE RAW DATA
# ============================================================

data.to_csv(
    "outputs/nifty50_raw_data.csv"
)

print("\nRaw data saved:")
print("outputs/nifty50_raw_data.csv")


# ============================================================
# 8. ADAPTIVE KALMAN FILTER
# ============================================================

def adaptive_kalman_filter(
    prices,
    Q=1.0,
    R=10.0
):
    """
    Simple causal Adaptive Kalman Filter.

    Q = process noise
    R = measurement noise

    The filter processes observations sequentially,
    so future values are not used to calculate past values.
    """

    prices = np.asarray(prices, dtype=float)

    n = len(prices)

    filtered = np.zeros(n)

    # Initial state
    x = prices[0]

    # Initial covariance
    P = 1.0

    for i in range(n):

        measurement = prices[i]

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        x_pred = x
        P_pred = P + Q

        # ----------------------------------------------------
        # Innovation
        # ----------------------------------------------------

        innovation = measurement - x_pred

        # ----------------------------------------------------
        # Adaptive measurement noise
        # ----------------------------------------------------

        adaptive_R = R + 0.1 * (
            innovation ** 2
        )

        # ----------------------------------------------------
        # Kalman Gain
        # ----------------------------------------------------

        K = P_pred / (
            P_pred + adaptive_R
        )

        # ----------------------------------------------------
        # Update
        # ----------------------------------------------------

        x = x_pred + K * innovation

        P = (1 - K) * P_pred

        filtered[i] = x

    return filtered


# ============================================================
# 9. APPLY ADAPTIVE KALMAN FILTER
# ============================================================

data["Filtered_Close"] = adaptive_kalman_filter(
    data["Close"].values
)


# ============================================================
# 10. TECHNICAL INDICATORS
# ============================================================

# ------------------------------------------------------------
# EMA 9
# ------------------------------------------------------------

data["EMA_9"] = (
    data["Close"]
    .ewm(span=9, adjust=False)
    .mean()
)


# ------------------------------------------------------------
# EMA 21
# ------------------------------------------------------------

data["EMA_21"] = (
    data["Close"]
    .ewm(span=21, adjust=False)
    .mean()
)


# ------------------------------------------------------------
# RSI 14
# ------------------------------------------------------------

delta = data["Close"].diff()

gain = delta.clip(lower=0)

loss = -delta.clip(upper=0)

avg_gain = gain.rolling(
    window=14,
    min_periods=14
).mean()

avg_loss = loss.rolling(
    window=14,
    min_periods=14
).mean()

rs = avg_gain / avg_loss.replace(
    0,
    np.nan
)

data["RSI_14"] = (
    100 - (
        100 / (1 + rs)
    )
)

# Handle cases where average loss is zero
data.loc[
    (avg_loss == 0) & (avg_gain > 0),
    "RSI_14"
] = 100

data.loc[
    (avg_gain == 0) & (avg_loss > 0),
    "RSI_14"
] = 0


# # ------------------------------------------------------------
# # ATR 14
# # ------------------------------------------------------------

# previous_close = data["Close"].shift(1)

# true_range_1 = (
#     data["High"] - data["Low"]
# )

# true_range_2 = (
#     data["High"] - previous_close
# ).abs()

# true_range_3 = (
#     data["Low"] - previous_close
# ).abs()

# true_range = pd.concat(
#     [
#         true_range_1,
#         true_range_2,
#         true_range_3
#     ],
#     axis=1
# ).max(axis=1)

# data["ATR_14"] = (
#     true_range
#     .rolling(
#         window=14,
#         min_periods=14
#     )
#     .mean()
# )


# data["ATR_14"] = (
#     true_range
#     .rolling(
#         window=14,
#         min_periods=14
#     )
#     .mean()
# )




# ------------------------------------------------------------
# ATR 14
# ------------------------------------------------------------

previous_close = data["Close"].shift(1)

true_range_1 = (
    data["High"] - data["Low"]
)

true_range_2 = (
    data["High"] - previous_close
).abs()

true_range_3 = (
    data["Low"] - previous_close
).abs()

true_range = pd.concat(
    [
        true_range_1,
        true_range_2,
        true_range_3
    ],
    axis=1
).max(axis=1)

data["ATR_14"] = (
    true_range
    .rolling(
        window=14,
        min_periods=14
    )
    .mean()
)


# ------------------------------------------------------------
# LOG RETURN (new prediction target)
# ------------------------------------------------------------
data["Log_Return"] = np.log(
    data["Close"] / data["Close"].shift(1)
)


# ============================================================
# 11. REMOVE WARM-UP NaN ROWS
# ============================================================

data = data.dropna().copy()


print("\n" + "=" * 70)
print("FEATURE DATASET")
print("=" * 70)

print("Rows after indicators:", len(data))

print("\nColumns:")
print(data.columns.tolist())


# ============================================================
# 12. FEATURES AND TARGET
# ============================================================

FEATURE_COLUMNS = [
    "Filtered_Close",
    "EMA_9",
    "EMA_21",
    "RSI_14",
    "ATR_14"
]

# TARGET_COLUMN = "Close"
TARGET_COLUMN = "Log_Return"


# ============================================================
# 13. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

n_rows = len(data)

train_end = int(
    n_rows * TRAIN_RATIO
)

validation_end = int(
    n_rows *
    (TRAIN_RATIO + VALIDATION_RATIO)
)

train_dates = data.index[:train_end]

validation_dates = data.index[
    train_end:validation_end
]

test_dates = data.index[
    validation_end:
]

print("\n" + "=" * 70)
print("CHRONOLOGICAL SPLIT")
print("=" * 70)

print(
    f"Train      : {len(train_dates)} rows"
)

print(
    f"Validation : {len(validation_dates)} rows"
)

print(
    f"Test       : {len(test_dates)} rows"
)

print(
    f"\nTrain start      : {train_dates.min()}"
)

print(
    f"Train end        : {train_dates.max()}"
)

print(
    f"Validation start : {validation_dates.min()}"
)

print(
    f"Validation end   : {validation_dates.max()}"
)

print(
    f"Test start       : {test_dates.min()}"
)

print(
    f"Test end         : {test_dates.max()}"
)


# ============================================================
# 14. FIT SCALERS ONLY ON TRAINING DATA
# ============================================================

feature_scaler = MinMaxScaler()

target_scaler = MinMaxScaler()


# Fit feature scaler using TRAINING DATA ONLY

feature_scaler.fit(
    data.loc[
        train_dates,
        FEATURE_COLUMNS
    ]
)


# Fit target scaler using TRAINING TARGET ONLY

target_scaler.fit(
    data.loc[
        train_dates,
        [TARGET_COLUMN]
    ]
)


# ============================================================
# 15. TRANSFORM FEATURES AND TARGET
# ============================================================

scaled_features = feature_scaler.transform(
    data[FEATURE_COLUMNS]
)

scaled_target = target_scaler.transform(
    data[[TARGET_COLUMN]]
)


scaled_features = pd.DataFrame(
    scaled_features,
    index=data.index,
    columns=FEATURE_COLUMNS
)

scaled_target = pd.Series(
    scaled_target.flatten(),
    index=data.index,
    name=TARGET_COLUMN
)


# ============================================================
# 16. CREATE TIME SERIES SEQUENCES
# ============================================================

def create_sequences(
    features,
    target,
    dates,
    sequence_length
):

    X = []
    y = []
    target_dates = []

    for i in range(
        sequence_length,
        len(features)
    ):

        X.append(
            features[
                i - sequence_length:i
            ]
        )

        y.append(
            target[i]
        )

        target_dates.append(
            dates[i]
        )

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        pd.DatetimeIndex(target_dates)
    )


X_all, y_all, sequence_dates = create_sequences(
    scaled_features.values,
    scaled_target.values,
    data.index,
    SEQ_LEN
)


print("\n" + "=" * 70)
print("SEQUENCE DATA")
print("=" * 70)

print("X shape:", X_all.shape)

print("y shape:", y_all.shape)


# ============================================================
# 17. SPLIT SEQUENCES BY TARGET DATE
# ============================================================

train_mask = (
    sequence_dates <= train_dates.max()
)

validation_mask = (
    (sequence_dates > train_dates.max()) &
    (sequence_dates <= validation_dates.max())
)

test_mask = (
    sequence_dates > validation_dates.max()
)


X_train = X_all[train_mask]

y_train = y_all[train_mask]


X_validation = X_all[validation_mask]

y_validation = y_all[validation_mask]


X_test = X_all[test_mask]

y_test = y_all[test_mask]


dates_train = sequence_dates[train_mask]

dates_validation = sequence_dates[
    validation_mask
]

dates_test = sequence_dates[
    test_mask
]


print("\nTraining sequences   :", len(X_train))

print("Validation sequences:", len(X_validation))

print("Test sequences      :", len(X_test))


# ============================================================
# 18. CHECK DATA
# ============================================================

if len(X_train) == 0:
    raise ValueError(
        "Training sequences are empty."
    )

if len(X_validation) == 0:
    raise ValueError(
        "Validation sequences are empty."
    )

if len(X_test) == 0:
    raise ValueError(
        "Test sequences are empty."
    )


# ============================================================
# 19. DCWRNN CELL
# ============================================================

@tf.keras.utils.register_keras_serializable()
class DCWRNNCell(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        units,
        update_interval,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.units = units

        self.update_interval = update_interval


    def build(
        self,
        input_shape
    ):

        self.W_h = self.add_weight(
            shape=(
                self.units,
                self.units
            ),
            initializer="glorot_uniform",
            name="W_h"
        )

        self.W_x = self.add_weight(
            shape=(
                input_shape[-1],
                self.units
            ),
            initializer="glorot_uniform",
            name="W_x"
        )

        self.b = self.add_weight(
            shape=(
                self.units,
            ),
            initializer="zeros",
            name="b"
        )

        super().build(input_shape)


    def call(
        self,
        x,
        h_prev,
        t
    ):

        should_update = tf.equal(
            tf.math.floormod(
                t,
                self.update_interval
            ),
            0
        )

        h_tilde = tf.nn.tanh(
            tf.matmul(
                h_prev,
                self.W_h
            )
            +
            tf.matmul(
                x,
                self.W_x
            )
            +
            self.b
        )

        return tf.where(
            should_update,
            h_tilde,
            h_prev
        )


# ============================================================
# 20. DCWRNN MODEL
# ============================================================

@tf.keras.utils.register_keras_serializable()
class DCWRNN(
    tf.keras.Model
):

    def __init__(
        self,
        units=32,
        update_intervals=None,
        **kwargs
    ):

        super().__init__(**kwargs)

        if update_intervals is None:
            update_intervals = [1, 3, 5]

        self.units = units

        self.update_intervals = (
            update_intervals
        )

        self.cells = [
            DCWRNNCell(
                units=units,
                update_interval=interval
            )
            for interval in update_intervals
        ]

        self.dense = tf.keras.layers.Dense(
            1
        )


    def call(
        self,
        inputs
    ):

        batch_size = tf.shape(
            inputs
        )[0]

        # One hidden state for every interval

        hidden_states = [
            tf.zeros(
                (
                    batch_size,
                    self.units
                ),
                dtype=tf.float32
            )
            for _ in self.cells
        ]


        # Sequence length is fixed at 5
        # which is our SEQ_LEN.

        sequence_length = inputs.shape[1]


        for t in range(sequence_length):

            x_t = inputs[:, t, :]

            new_hidden_states = []

            for i, cell in enumerate(
                self.cells
            ):

                h = cell(
                    x_t,
                    hidden_states[i],
                    tf.constant(
                        t,
                        dtype=tf.int32
                    )
                )

                new_hidden_states.append(h)

            hidden_states = (
                new_hidden_states
            )


        # Combine hidden states

        combined = tf.concat(
            hidden_states,
            axis=1
        )


        # Final prediction

        output = self.dense(
            combined
        )

        return output


# ============================================================
# 21. CREATE MODEL
# ============================================================

model = DCWRNN(
    units=DCWRNN_UNITS,
    update_intervals=UPDATE_INTERVALS
)


# ============================================================
# 22. BUILD MODEL
# ============================================================

dummy_input = tf.zeros(
    (
        1,
        SEQ_LEN,
        len(FEATURE_COLUMNS)
    )
)

dummy_output = model(
    dummy_input
)


print("\n" + "=" * 70)
print("MODEL")
print("=" * 70)

model.summary()


# ============================================================
# 23. COMPILE MODEL
# ============================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss="mse",
    metrics=["mae"]
)


# ============================================================
# 24. CALLBACKS
# ============================================================

checkpoint_path = (
    "models/best_dcwrnn.weights.h5"
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
    mode="min",
    verbose=1
)


early_stopping = (
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        mode="min",
        verbose=1
    )
)


# ============================================================
# 25. TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING DCWRNN")
print("=" * 70)

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_validation,
        y_validation
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        checkpoint,
        early_stopping
    ],
    shuffle=False,
    verbose=1
)


# ============================================================
# 26. LOAD BEST MODEL WEIGHTS
# ============================================================

if os.path.exists(
    checkpoint_path
):

    model.load_weights(
        checkpoint_path
    )

    print(
        "\nBest DCWRNN weights loaded."
    )


# ============================================================
# 27. PREDICT TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("TEST PREDICTION")
print("=" * 70)

predicted_scaled = model.predict(
    X_test,
    verbose=0
)


# ============================================================
# 28. CONVERT PREDICTIONS BACK TO ORIGINAL PRICE
# ============================================================

# predicted_prices = (
#     target_scaler
#     .inverse_transform(
#         predicted_scaled
#     )
#     .flatten()
# )


# actual_prices = (
#     data.loc[
#         dates_test,
#         TARGET_COLUMN
#     ]
#     .values
# )


predicted_log_returns = (
    target_scaler
    .inverse_transform(
        predicted_scaled
    )
    .flatten()
)

previous_closes_test = np.array([
    data["Close"].iloc[data.index.get_loc(d) - 1]
    for d in dates_test
])

predicted_prices = previous_closes_test * np.exp(predicted_log_returns)

actual_prices = (
    data.loc[
        dates_test,
        "Close"
    ]
    .values
)

actual_log_returns = np.log(actual_prices / previous_closes_test)


# ============================================================
# 29. NAIVE BASELINE
# ============================================================

# Naive prediction:
# tomorrow's price = today's actual price

naive_predictions = []

naive_actuals = []


for target_date in dates_test:

    position = data.index.get_loc(
        target_date
    )

    if position == 0:
        continue

    previous_close = data[
        "Close"
    ].iloc[position - 1]

    actual_close = data[
        "Close"
    ].iloc[position]

    naive_predictions.append(
        previous_close
    )

    naive_actuals.append(
        actual_close
    )


naive_predictions = np.array(
    naive_predictions
)

naive_actuals = np.array(
    naive_actuals
)


# ============================================================
# 30. MODEL METRICS
# ============================================================

mae = mean_absolute_error(
    actual_prices,
    predicted_prices
)

mse = mean_squared_error(
    actual_prices,
    predicted_prices
)

rmse = np.sqrt(mse)

r2 = r2_score(
    actual_prices,
    predicted_prices
)


# ============================================================
# 31. NAIVE BASELINE METRICS
# ============================================================

naive_mae = mean_absolute_error(
    naive_actuals,
    naive_predictions
)

naive_mse = mean_squared_error(
    naive_actuals,
    naive_predictions
)

naive_rmse = np.sqrt(
    naive_mse
)

naive_r2 = r2_score(
    naive_actuals,
    naive_predictions
)


# ============================================================
# 32. PRINT METRICS
# ============================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"DCWRNN MAE  : {mae:.4f}"
)

print(
    f"DCWRNN MSE  : {mse:.4f}"
)

print(
    f"DCWRNN RMSE : {rmse:.4f}"
)

print(
    f"DCWRNN R2   : {r2:.4f}"
)


direction_correct = np.sign(predicted_log_returns) == np.sign(actual_log_returns)
directional_accuracy = direction_correct.mean() * 100
pct_up_days = (actual_log_returns > 0).mean() * 100

print(f"\nDCWRNN Directional Accuracy: {directional_accuracy:.2f}%")
print(f"Baseline (always predict UP)  : {pct_up_days:.2f}%")
print(f"Baseline (always predict DOWN): {100 - pct_up_days:.2f}%")




print("\n" + "-" * 70)
print("NAIVE BASELINE")
print("-" * 70)

print(
    f"Naive MAE  : {naive_mae:.4f}"
)

print(
    f"Naive MSE  : {naive_mse:.4f}"
)

print(
    f"Naive RMSE : {naive_rmse:.4f}"
)

print(
    f"Naive R2   : {naive_r2:.4f}"
)


# ============================================================
# 33. CREATE COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame(
    {
        "Metric": [
            "MAE",
            "MSE",
            "RMSE",
            "R2"
        ],

        "DCWRNN": [
            mae,
            mse,
            rmse,
            r2
        ],

        "Naive_Baseline": [
            naive_mae,
            naive_mse,
            naive_rmse,
            naive_r2
        ]
    }
)


comparison.to_csv(
    "outputs/model_comparison.csv",
    index=False
)


# ============================================================
# 34. SAVE TEST PREDICTIONS
# ============================================================

prediction_results = pd.DataFrame(
    {
        "Date": dates_test,

        "Actual_Close": actual_prices,

        "Predicted_Close": predicted_prices
    }
)


prediction_results.to_csv(
    "signals/dcwrnn_signals.csv",
    index=False
)


prediction_results.to_csv(
    "outputs/dcwrnn_predictions.csv",
    index=False
)


print("\nPrediction files saved:")

print(
    "signals/dcwrnn_signals.csv"
)

print(
    "outputs/dcwrnn_predictions.csv"
)


# ============================================================
# 35. TRAINING HISTORY CSV
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    "outputs/training_history.csv",
    index=False
)


# ============================================================
# 36. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(
    {
        "Metric": [
            "MAE",
            "MSE",
            "RMSE",
            "R2"
        ],

        "DCWRNN": [
            mae,
            mse,
            rmse,
            r2
        ],

        "Naive_Baseline": [
            naive_mae,
            naive_mse,
            naive_rmse,
            naive_r2
        ]
    }
)


metrics_df.to_csv(
    "outputs/metrics.csv",
    index=False
)


# ============================================================
# 37. GRAPH 1 - TRAINING LOSS
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "DCWRNN Training and Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Mean Squared Error"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "graphs/training_validation_loss.png",
    dpi=300
)

plt.close()


# ============================================================
# 38. GRAPH 2 - ACTUAL VS PREDICTED
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    dates_test,
    actual_prices,
    label="Actual Close"
)

plt.plot(
    dates_test,
    predicted_prices,
    label="DCWRNN Predicted Close"
)

plt.title(
    "NIFTY 50 Actual vs DCWRNN Predicted Close"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "NIFTY 50 Close Price"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    "graphs/actual_vs_predicted.png",
    dpi=300
)

plt.close()


# ============================================================
# 39. GRAPH 3 - DCWRNN VS NAIVE
# ============================================================

# Make lengths equal for plotting

naive_plot = []

for target_date in dates_test:

    position = data.index.get_loc(
        target_date
    )

    if position == 0:
        naive_plot.append(
            np.nan
        )
    else:
        naive_plot.append(
            data[
                "Close"
            ].iloc[position - 1]
        )


naive_plot = np.array(
    naive_plot
)


plt.figure(
    figsize=(12, 6)
)

plt.plot(
    dates_test,
    actual_prices,
    label="Actual Close"
)

plt.plot(
    dates_test,
    predicted_prices,
    label="DCWRNN"
)

plt.plot(
    dates_test,
    naive_plot,
    label="Naive Baseline"
)

plt.title(
    "DCWRNN vs Naive Baseline"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "NIFTY 50 Close Price"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    "graphs/dcwrnn_vs_naive.png",
    dpi=300
)

plt.close()


# ============================================================
# 40. GRAPH 4 - ACTUAL PRICE
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    data.index,
    data["Close"],
    label="NIFTY 50 Close"
)

plt.title(
    "NIFTY 50 Historical Close Price"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Close Price"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    "graphs/nifty50_close_price.png",
    dpi=300
)

plt.close()


# ============================================================
# 41. GRAPH 5 - KALMAN FILTER
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    data.index,
    data["Close"],
    label="Original Close"
)

plt.plot(
    data.index,
    data["Filtered_Close"],
    label="Adaptive Kalman Filter"
)

plt.title(
    "NIFTY 50 Close Price with Adaptive Kalman Filtering"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Price"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    "graphs/kalman_filtered_close.png",
    dpi=300
)

plt.close()








# ============================================================
# 42. GENERATE NEXT-DAY PREDICTION
# ============================================================

latest_sequence = scaled_features.iloc[
    -SEQ_LEN:
].values.astype(
    np.float32
)

latest_sequence = np.expand_dims(
    latest_sequence,
    axis=0
)

next_day_scaled = model.predict(
    latest_sequence,
    verbose=0
)

# ============================================================
# 42. GENERATE NEXT-DAY PREDICTION
# ============================================================

predicted_log_return = (
    target_scaler
    .inverse_transform(
        next_day_scaled
    )
    .flatten()[0]
)

last_close = data["Close"].iloc[-1]

next_day_prediction = last_close * np.exp(predicted_log_return)

predicted_change = (
    next_day_prediction -
    last_close
)

predicted_change_percent = (
    predicted_change /
    last_close
) * 100


# ============================================================
# 43. NEXT TRADING DATE
# ============================================================

last_date = data.index[-1]

next_date = (
    last_date +
    pd.Timedelta(days=1)
)

while next_date.weekday() >= 5:

    next_date += pd.Timedelta(
        days=1
    )


# ============================================================
# 44. SAVE NEXT-DAY PREDICTION
# ============================================================

next_day_result = pd.DataFrame(
    {
        "Last_Date": [
            last_date
        ],

        "Prediction_Date": [
            next_date
        ],

        "Last_Close": [
            last_close
        ],

        "Predicted_Close": [
            next_day_prediction
        ],

        "Predicted_Change": [
            predicted_change
        ],

        "Predicted_Change_Percent": [
            predicted_change_percent
        ]
    }
)


next_day_result.to_csv(
    "outputs/next_day_prediction.csv",
    index=False
)


# ============================================================
# 45. PRINT NEXT-DAY RESULT
# ============================================================

print("\n" + "=" * 70)
print("NEXT TRADING-DAY PREDICTION")
print("=" * 70)

print(
    "Last available date:",
    last_date.date()
)

print(
    "Prediction date:",
    next_date.date()
)

print(
    f"Last Close: ₹{last_close:.2f}"
)

print(
    f"Predicted Close: ₹{next_day_prediction:.2f}"
)

print(
    f"Predicted Change: ₹{predicted_change:.2f}"
)

print(
    f"Predicted Change %: "
    f"{predicted_change_percent:.2f}%"
)


if predicted_change_percent > 0:

    print(
        "Prediction Direction: UP"
    )

elif predicted_change_percent < 0:

    print(
        "Prediction Direction: DOWN"
    )

else:

    print(
        "Prediction Direction: NO CHANGE"
    )


# ============================================================
# 46. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 0 COMPLETED")
print("=" * 70)

print("\nFiles created:")

print("\nMODEL:")
print(
    "models/best_dcwrnn.weights.h5"
)

print("\nRAW DATA:")
print(
    "outputs/nifty50_raw_data.csv"
)

print("\nPREDICTIONS:")
print(
    "signals/dcwrnn_signals.csv"
)

print(
    "outputs/dcwrnn_predictions.csv"
)

print("\nMETRICS:")
print(
    "outputs/metrics.csv"
)

print(
    "outputs/model_comparison.csv"
)

print("\nNEXT-DAY:")
print(
    "outputs/next_day_prediction.csv"
)

print("\nTRAINING:")
print(
    "outputs/training_history.csv"
)

print("\nGRAPHS:")
print(
    "graphs/training_validation_loss.png"
)

print(
    "graphs/actual_vs_predicted.png"
)

print(
    "graphs/dcwrnn_vs_naive.png"
)

print(
    "graphs/nifty50_close_price.png"
)

print(
    "graphs/kalman_filtered_close.png"
)

print("\n" + "=" * 70)
print("READY FOR PHASE 1 BACKTESTING")
print("=" * 70)




from scipy.stats import binomtest

n_correct = int(round(directional_accuracy / 100 * len(dates_test)))
sig_test = binomtest(n_correct, len(dates_test), 0.5, alternative='greater')
print(f"Binomial test vs. 50% chance: p = {sig_test.pvalue:.4f}")