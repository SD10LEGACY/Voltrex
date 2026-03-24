print("--- PHASE 1: LIVE BINANCE DATA + HYBRID v8 ---")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from binance.client import Client
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")
plt.style.use('dark_background')
LOOK_BACK = 60

# --- 1. LIVE DATA INGESTION (Replaces CSV) ---
print("Connecting to Binance API...")
client = Client() # No API keys needed for public price data

print("Downloading live BTC/USDT daily data from 2018 to Today...")
# Fetch daily klines (candlesticks)
klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1DAY, "1 Jan, 2018", "today UTC")

# Convert Binance data into a Pandas DataFrame
cols = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base', 'Taker buy quote', 'Ignore']
df = pd.DataFrame(klines, columns=cols)

# Convert strings to floats
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)

# Fix Datetime exactly like the old CSV
df['Open time'] = pd.to_datetime(df['Open time'], unit='ms', utc=True)
data_df = df.set_index('Open time').copy()

# --- 2. ADVANCED FEATURE ENGINEERING ---
print("Generating Advanced Technical Indicators...")
delta = data_df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data_df['RSI'] = 100 - (100 / (1 + rs))

ema12 = data_df['Close'].ewm(span=12, adjust=False).mean()
ema26 = data_df['Close'].ewm(span=26, adjust=False).mean()
data_df['MACD'] = ema12 - ema26
data_df['MACD_Signal'] = data_df['MACD'].ewm(span=9, adjust=False).mean()

data_df['BB_Mid'] = data_df['Close'].rolling(20).mean()
data_df['BB_Upper'] = data_df['BB_Mid'] + (data_df['Close'].rolling(20).std() * 2)
data_df['BB_Lower'] = data_df['BB_Mid'] - (data_df['Close'].rolling(20).std() * 2)

high_low = data_df['High'] - data_df['Low']
high_close = np.abs(data_df['High'] - data_df['Close'].shift())
low_close = np.abs(data_df['Low'] - data_df['Close'].shift())
ranges = pd.concat([high_low, high_close, low_close], axis=1)
data_df['ATR'] = np.max(ranges, axis=1).rolling(14).mean()

data_df['OBV'] = (np.sign(data_df['Close'].diff()) * data_df['Volume']).fillna(0).cumsum()

data_df = data_df.dropna()
print(f"Live Data Prepared! Shape: {data_df.shape}")

# --- 3. SCALE & SEQUENCE ---
target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaled = target_scaler.fit_transform(data_df[['Close']])

feature_cols = ['Close', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'ATR', 'OBV']
feature_scaler = RobustScaler()
features_scaled = feature_scaler.fit_transform(data_df[feature_cols])

def create_sequences(features, target, look_back=60):
    X, y = [], []
    for i in range(len(features) - look_back - 1): 
        X.append(features[i:(i + look_back)])
        y.append(target[i + look_back]) 
    return np.array(X), np.array(y)

X_all, y_all = create_sequences(features_scaled, target_scaled, LOOK_BACK)

train_size = int(len(X_all) * 0.95)
train_X, train_y = X_all[:train_size], y_all[:train_size]
test_X, test_y = X_all[train_size:], y_all[train_size:]

# --- 4. TRAIN LSTM ---
print("Training Neural Network (Stage 1)...")
input_layer = Input(shape=(LOOK_BACK, train_X.shape[2]))
x = LSTM(128, return_sequences=True)(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = LSTM(64, return_sequences=False)(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
output_layer = Dense(1)(x)

model_lstm = Model(inputs=input_layer, outputs=output_layer)
model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model_lstm.fit(train_X, train_y, epochs=30, batch_size=32, validation_data=(test_X, test_y), callbacks=[early_stop], verbose=0, shuffle=False)

# --- 5. TRAIN XGBOOST ---
print("Extracting Deep Features & Training XGBoost (Stage 2)...")
intermediate_model = Model(inputs=model_lstm.input, outputs=model_lstm.layers[-4].output)
X_train_hybrid = np.concatenate([intermediate_model.predict(train_X, verbose=0), train_X[:, -1, :]], axis=1)
X_test_hybrid = np.concatenate([intermediate_model.predict(test_X, verbose=0), test_X[:, -1, :]], axis=1)

xgb_model = XGBRegressor(n_estimators=2000, learning_rate=0.005, max_depth=7, subsample=0.7, colsample_bytree=0.7, n_jobs=-1, early_stopping_rounds=50)
xgb_model.fit(X_train_hybrid, train_y, eval_set=[(X_test_hybrid, test_y)], verbose=False)

# --- 6. LIVE PREDICTION ---
print("Forecasting Next Day based on LIVE market data...")
last_seq_features = features_scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, features_scaled.shape[1])
hybrid_feat_tomorrow = np.concatenate([intermediate_model.predict(last_seq_features, verbose=0), features_scaled[-1].reshape(1, -1)], axis=1)

pred_tomorrow = target_scaler.inverse_transform(xgb_model.predict(hybrid_feat_tomorrow).reshape(-1, 1))[0][0]

last_date = data_df.index[-1]
next_date = last_date + pd.Timedelta(days=1)

print("=" * 50)
print(f"OFFICIAL LIVE PREDICTION FOR ({next_date.date()})")
print(f"Current Binance Last Price: ${data_df['Close'].iloc[-1]:.2f}")
print(f"HYBRID v8 PREDICTION: ${pred_tomorrow:.2f}")
print("=" * 50)