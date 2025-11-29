```python
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib
import pandas_ta as ta

# Moon Dev Data Preparation 🌙
def load_data(path):
    data = pd.read_csv(path)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Drop unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    # Proper column mapping
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'datetime': 'Date'
    }, inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

class DivergenceDecay(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌙
    atr_multiplier = 3  # 3x ATR for stops 🛑
    lookback_period = 20  # New high detection period 📈
    
    def init(self):
        # Moon Dev Indicator Setup 🌙✨
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.high_20 = self.I(talib.MAX, self.data.High, self.lookback_period)
        
        # Track previous peaks for divergence detection
        self.prev_peak_idx = None
        self.prev_peak_obv = -np.inf
        
    def next(self):
        # Moon Dev Strategy Logic 🌙➡️✨
        if len(self.data) < self.lookback_period + 3:  # Warmup period
            return
            
        current_high = self.data.High[-1]
        current_obv = self.obv[-1]
        high_20 = self.high_20[-1]
        
        # 1. Price 新高 Detection 🌕
        new_high_trigger = (current_high == high_20)
        
        # 2. Bearish OBV Divergence 🔻
        obv_divergence = False
        if new_high_trigger and self.prev_peak_idx is not None:
            obv_at_prev_peak = self.obv[self.prev_peak_idx]
            obv_divergence = current_obv < obv_at_prev_peak
        
        # 3. Declining ATR Condition 📉
        atr_declining = False
        if len(self.atr) >= 3:
            atr_declining = (self.atr[-1] < self.atr[-2]) or (self.atr[-1] < self.atr[-3])
        
        # Moon Dev Entry Logic 🌙🎯
        if not self.position and new_high_trigger and obv_divergence and atr_declining:
            # Risk Management Calculations 🧮
            atr_value = self.atr[-1]
            entry_price = self.data.Close[-1]
            risk_amount = self.equity * self.risk_percent
            risk_per_share = self.atr_multiplier * atr_value
            
            # Position Sizing Rule ⚖️
            position_size = risk_amount / risk_per_share
            position_size = int(round(position_size))
            
            if position_size > 0:
                self.trailing_stop = entry_price + self.atr_multiplier * atr_value
                self.sell(size=position_size)
                print(f"🌙✨ Moon Dev Short Signal! ✨")
                print(f"Entry: {entry_price:.2f} | Size: {position_size}")
                print(f"Initial Stop: {self.trailing_stop:.2f} | ATR: {atr_value:.2f}")
        
        # Moon Dev Exit Management 🌙🛑
        if self.position:
            # Update trailing stop with current ATR
            current_atr = self.atr[-1]
            new_stop = self.data.High[-1] + self.atr_multiplier * current_atr
            self.trailing_stop = min(self.trailing_stop, new_stop)
            
            # Check stop breach
            if self.data.High[-1] >= self.trailing_stop:
                self.position.close()
                print(f"🚨🌑 Moon Dev Exit: Trailing Stop {self.trailing