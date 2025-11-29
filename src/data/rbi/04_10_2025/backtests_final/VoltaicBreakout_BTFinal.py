Here's the complete fixed code with Moon Dev themed debug prints and proper position sizing:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Strategy, Backtest
import numpy as np

# Moon Dev Data Preparation 🌙
def load_data(path):
    print("🌙 Loading cosmic market data...")
    data = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Drop unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    # Map to proper case
    column_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    data = data.rename(columns=column_mapping)
    print("🌙✨ Data successfully aligned with lunar cycles!")
    return data

class VoltaicBreakout(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # Moon Dev Indicator Initialization ✨
        print("🌙 Initializing Moon Dev Indicators...")
        
        # Calculate indicators using TA-Lib through self.I()
        self.cmo = self.I(talib.CMO, self.data.Close, timeperiod=14, name='CMO')
        self.cmo_sma = self.I(talib.SMA, self.cmo, timeperiod=20, name='CMO_SMA')
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=10, name='Volume_SMA')
        
        # Keltner Channel components
        typical_price = (self.data.High + self.data.Low + self.data.Close) / 3
        self.ema20 = self.I(talib.EMA, typical_price, timeperiod=20, name='EMA_20')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=20, name='ATR_20')
        
        print("🌙✨ Indicators successfully initialized!")

    def next(self):
        current_ema = self.ema20[-1]
        current_atr = self.atr[-1]
        current_upper = current_ema + 2 * current_atr
        current_lower = current_ema - 2 * current_atr
        
        # Moon Dev Exit Logic 🌙✨
        if self.position.is_long and self.data.Low[-1] <= current_lower:
            print(f"🌙✨ Closing LONG at {self.data.Close[-1]:.2f} (Trailing Stop Hit!)")
            self.position.close()
            
        if self.position.is_short and self.data.High[-1] >= current_upper:
            print(f"🌙✨ Closing SHORT at {self.data.Close[-1]:.2f} (Trailing Stop Hit!)")
            self.position.close()
        
        # Skip if already in position
        if self.position:
            return
            
        # Moon Dev Entry Logic 🚀
        current_cmo = self.cmo[-1]
        current_sma = self.cmo_sma[-1]
        prev_cmo = self.cmo[-2] if len(self.cmo) > 1 else np.nan
        prev_sma = self.cmo_sma[-2] if len(self.cmo_sma) > 1 else np.nan
        
        volume_condition = self.data.Volume[-1] > 2 * self.volume_sma[-1]
        
        # Long Entry (Bullish Crossover)
        if (prev_cmo <= prev_sma) and (current_cmo > current_sma) and volume_condition:
            risk = self.data.Close[-1] - current_lower
            if risk <= 0:
                print("🌙⚠️ Invalid long risk - skipping trade")
                return
            size = int(round((self.equity * self.risk_percent) / risk))
            print(f"🚀🌙 BULLISH BREAKOUT! Buying {size} units @ {self.data.Close[-1]:.2f}")
            self.buy(size=size, sl=current_lower)
            
        # Short Entry (Bearish Crossover)
        elif (prev_cmo >= prev_sma) and (current_cmo < current_sma) and volume_condition:
            risk = current_upper - self.data.Close[-1]
            if risk <= 0:
                print("🌙⚠️ Invalid short risk - skipping trade")
                return
            size = int(round((self.e