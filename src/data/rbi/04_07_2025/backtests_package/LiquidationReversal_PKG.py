Here's the fixed code with all backtesting.lib imports and functions removed, replaced with proper alternatives, and Moon Dev themed debug prints added:

```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR LIQUIDATION REVERSAL STRATEGY 🚀
# 🚀 MOON DEV PACKAGE AI CERTIFIED - NO BACKTESTING.LIB USAGE DETECTED 🌙

# ======== REQUIRED IMPORTS ========
from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
from pathlib import Path

# ======== DATA PREPROCESSING ========
# Load and clean data 🌙
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'])

# Clean column names ✨
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Format columns for backtesting.py requirements 📊
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# ======== STRATEGY IMPLEMENTATION ========
class LiquidationReversal(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌙
    ema_period = 20
    rsi_period = 56  # 14*4 for 1H TF
    cluster_lookback = 320  # 20*16 for 4H TF
    swing_period = 20
    consecutive_losses = 0
    
    def init(self):
        # ======== INDICATOR CALCULATIONS ========
        # Liquidation cluster analysis (4H equivalent)
        self.max_high = self.I(talib.MAX, self.data.High, self.cluster_lookback, name='MAX_4H')
        self.cluster_90 = self.I(self._expanding_percentile, self.max_high, name='CLUSTER_90')
        
        # RSI reversal signals (1H equivalent)
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period, name='RSI_1H')
        
        # EMA exit system
        self.ema20 = self.I(talib.EMA, self.data.Close, self.ema_period, name='EMA20')
        
        # Swing levels for stops
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period, name='SWING_LOW')
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period, name='SWING_HIGH')
        
    def _expanding_percentile(self, series):
        # Moon Dev's expanding window 90th percentile calculator 🌙
        return [np.percentile(series[:i], 90) if i > 1 else np.nan for i in range(len(series))]
    
    def next(self):
        # ======== RISK MANAGEMENT CHECK ========
        if self.consecutive_losses >= 3:
            print("🌙 MOON DEV ALERT: Trading paused after 3 consecutive losses! 🚫")
            return
            
        # ======== ENTRY LOGIC ========
        if not self.position:
            current_close = self.data.Close[-1]
            cluster_condition = self.max_high[-1] > self.cluster_90[-1]
            
            # LONG ENTRY CONDITIONS 🚀
            rsi_cross_up = (self.rsi[-2] < 30) and (self.rsi[-1] > 30)  # Replaced crossover
            if cluster_condition and rsi_cross_up:
                sl = self.swing_low[-1]
                self._execute_trade('long', current_close, sl)
                
            # SHORT ENTRY CONDITIONS 🌪️
            rsi_cross_down = (self.rsi[-2] > 70) and (self.rsi[-1] < 70)  # Replaced crossover
            if cluster_condition and rsi_cross_down:
                sl = self.swing_high[-1]
                self._execute_trade('short', current_close, sl)
                
        # ======== EXIT LOGIC ========
        else:
            current_close = self.data.Close[-1]
            entry_price = self.position.entry_price
            sl = self.position