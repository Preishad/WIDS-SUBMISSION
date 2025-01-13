import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
import os
import random

nifty_symbols = [
    "ADANIPORTS", "ASIANPAINT", "AXISBANK", "BAJAJFINSV",
    "BAJFINANCE", "BPCL", "BRITANNIA", "CIPLA", "COALINDIA",
    "DIVISLAB", "DRREDDY", "EICHERMOT", "GRASIM", "HCLTECH",
    "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR",
    "ICICIBANK", "ICICIGI", "IOC", "INDUSINDBK", "INFY",
    "ITC", "JSWSTEEL", "KOTAKBANK", "LTTS", "LT",
    "MARICO", "MARUTI", "NESTLEIND"
]

def simulate_stock_data(symbols, days=300):
    data = {}
    for symbol in symbols:
        np.random.seed(hash(symbol) % (2**32))  # Seed based on symbol
        base_price = random.uniform(100, 1500)
        close_prices = np.cumsum(np.random.normal(0, 5, days)) + base_price
        high_prices = close_prices + np.random.uniform(1, 10, days)
        low_prices = close_prices - np.random.uniform(1, 10, days)
        volume = np.random.randint(1000, 10000, days)  # Simulate volume
        data[symbol] = pd.DataFrame({
            'close': close_prices,
            'high': high_prices,
            'low': low_prices,
            'volume': volume
        })
    return data

def calculate_indicators(data):
    data['MACD'] = data['close'].ewm(span=12, adjust=False).mean() - data['close'].ewm(span=26, adjust=False).mean()
    data['RSI'] = 100 - (100 / (1 + data['close'].pct_change().apply(lambda x: (x + abs(x)) / 2).rolling(14).mean()))
    data['Bollinger_High'] = data['close'].rolling(20).mean() + 2 * data['close'].rolling(20).std()
    data['Bollinger_Low'] = data['close'].rolling(20).mean() - 2 * data['close'].rolling(20).std()
    data['ATR'] = (data['high'] - data['low']).rolling(14).mean()
    data['SMA_50'] = data['close'].rolling(50).mean()
    data['SMA_200'] = data['close'].rolling(200).mean()
    data['Volume_Change'] = data['volume'].pct_change()
    data['Price_Volume_Interaction'] = data['close'] * data['volume']
    data.fillna(0, inplace=True)  # Replace NaN values with 0
    return data

def prepare_data_with_indicators(data, window_size):
    features = data[['MACD', 'RSI', 'Bollinger_High', 'Bollinger_Low', 'ATR', 'SMA_50', 'SMA_200', 'volume', 'Volume_Change', 'Price_Volume_Interaction']]
    prices = data['close']
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    pct_change = prices.pct_change().shift(-1) * 100  # Shift for next day prediction
    pct_change_class = (pct_change > 0).astype(int)  # 1 for up, 0 for down

    X, y = [], []
    for i in range(window_size, len(data) - 1):
        X.append(features_scaled[i-window_size:i])
        y.append([pct_change.iloc[i], pct_change_class.iloc[i]])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler

# LSTM Model for Classification
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(256, activation='tanh', return_sequences=True, input_shape=(input_shape[0], input_shape[1])),
        BatchNormalization(),
        Dropout(0.5),
        LSTM(128, activation='tanh', return_sequences=True),
        BatchNormalization(),
        Dropout(0.5),
        LSTM(64, activation='tanh', return_sequences=False),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')  # Predicts percentage change class (up/down)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'mse'])
    return model

if __name__ == "__main__":
   
    stock_data = simulate_stock_data(nifty_symbols)

    symbol = "ADANIPORTS"
    raw_data = stock_data[symbol]
    data_with_indicators = calculate_indicators(raw_data)

    
    window_size = 50
    X, y, scaler = prepare_data_with_indicators(data_with_indicators, window_size)

 
    y_pct_change = y[:, 0]
    y_class = tf.keras.utils.to_categorical(y[:, 1])

    
    X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)
    
    model_lstm = create_lstm_model(X_train.shape[1:])
    model_lstm.fit(X_train, y_train_class, validation_data=(X_test, y_test_class), epochs=100, batch_size=32, verbose=1)

    last_data = X[-1:]  # Last window of data
    next_day_prediction = model_lstm.predict(last_data)
    next_day_class = np.argmax(next_day_prediction[0])  # 0 for down, 1 for up
    print(f"Next Day Predicted Class for {symbol}: {'Up' if next_day_class == 1 else 'Down'}")

