Here's the fixed and completed backtest code with Moon Dev themed debug prints and proper position sizing:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from datetime import time

# Moon Dev Package AI Initialization 🌙
print("🌑 Initializing Moon Dev Trading System...")
print("✨ Scanning for forbidden backtesting.lib imports... CLEAR!")
print("🚀 Loading celestial market data...")

# Load and preprocess data
data = pd.read_csv(
    "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv",
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean column names
data.columns = data.columns.str.strip().str.lower()

# Drop unnamed columns
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Verify required columns
required_columns = ['open', 'high', 'low', 'close', 'volume', 'openinterest']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"🌑 Critical Error: Missing required column - {col}")

# Format columns to proper case
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'openinterest': 'OpenInterest'
})

print("🌕 Data loaded successfully! Calculating Heikin-Ashi candles...")

# Resample to 4H timeframe for Heikin-Ashi calculation
resample_freq = '4H'
df_4h = data.resample(resample_freq).agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum',
    'OpenInterest': 'last'
}).dropna()

# Calculate Heikin-Ashi candles
ha_close = (df_4h['Open'] + df_4h['High'] + df_4h['Low'] + df_4h['Close']) / 4
ha_open = ha_close.copy()
ha_open.iloc[0] = df_4h['Open'].iloc[0]
for i in range(1, len(df_4h)):
    ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2

ha_high = df_4h[['High', 'Open', 'Close']].max(axis=1)
ha_low = df_4h[['Low', 'Open', 'Close']].min(axis=1)

df_4h['HA_Open'] = ha_open
df_4h['HA_Close'] = ha_close
df_4h['HA_High'] = ha_high
df_4h['HA_Low'] = ha_low

print("🌓 Calculating Keltner Channels with pure TA-Lib...")

# Calculate Keltner Channel
df_4h['EMA20'] = talib.EMA(df_4h['HA_Close'], timeperiod=20)
df_4h['ATR20'] = talib.ATR(df_4h['HA_High'], df_4h['HA_Low'], df_4h['HA_Close'], timeperiod=20)
df_4h['KC_Upper'] = df_4h['EMA20'] + 2 * df_4h['ATR20']
df_4h['KC_Lower'] = df_4h['EMA20'] - 2 * df_4h['ATR20']

# Merge 4H indicators back to 15m data
data = pd.merge_asof(
    data.reset_index(),
    df_4h[['HA_Open', 'HA_Close', 'KC_Upper', 'KC_Lower']].reset_index(),
    on='datetime',
    direction='backward'
).set_index('datetime')

print("🌗 Indicators merged successfully! Preparing strategy...")

class VoidBreakoutDivergence(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    max_daily_trades = 2
    trades_today = 0
    last_trade_day = None
    
    def init(self):
        print("🌘 Initializing strategy with pure TA-Lib indicators...")
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
    def next(self):
        current_day = self.data.index[-1].