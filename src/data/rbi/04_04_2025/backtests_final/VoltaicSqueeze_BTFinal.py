Here's the fixed and complete backtest code with Moon Dev themed improvements:

```python
from backtesting import Strategy, Backtest
import pandas as pd
import talib
import numpy as np

# Moon Dev Data Preparation 🌙✨
print("🌙 Initializing Moon Dev Data Processing...")
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean and format data columns
print("✨ Cleaning cosmic data dust...")
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert and set datetime index
print("🌌 Aligning temporal dimensions...")
data['DateTime'] = pd.to_datetime(data['datetime'])
data.set_index('DateTime', inplace=True)
data.sort_index(inplace=True)

# Moon Dev Indicator Functions 🌙📊
def bb_upper(close, timeperiod=20, nbdev=2):
    """Calculate Bollinger Band Upper with pure TA-Lib"""
    sma = talib.SMA(close, timeperiod)
    std = talib.STDDEV(close, timeperiod)
    return sma + nbdev * std

def bb_lower(close, timeperiod=20, nbdev=2):
    """Calculate Bollinger Band Lower with pure TA-Lib"""
    sma = talib.SMA(close, timeperiod)
    std = talib.STDDEV(close, timeperiod)
    return sma - nbdev * std

def calculate_bb_width(upper, lower):
    """Calculate BB Width without external dependencies"""
    return upper - lower

def kc_upper(high, low, close, ema_period=20, atr_period=14, multiplier=2):
    """Keltner Channel Upper with pure TA-Lib"""
    ema = talib.EMA(close, ema_period)
    atr = talib.ATR(high, low, close, atr_period)
    return ema + multiplier * atr

def kc_lower(high, low, close, ema_period=20, atr_period=14, multiplier=2):
    """Keltner Channel Lower with pure TA-Lib"""
    ema = talib.EMA(close, ema_period)
    atr = talib.ATR(high, low, close, atr_period)
    return ema - multiplier * atr

class VoltaicSqueeze(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Moon Dev Indicators Initialization 🌙⚡
        print("🌠 Initializing Moon Dev Indicators...")
        self.bb_upper = self.I(bb_upper, self.data.Close, 20)
        self.bb_lower = self.I(bb_lower, self.data.Close, 20)
        self.bb_width = self.I(calculate_bb_width, self.bb_upper, self.bb_lower)
        self.bb_width_ma = self.I(talib.SMA, self.bb_width, 10)
        
        self.kc_upper = self.I(kc_upper, self.data.High, self.data.Low, self.data.Close, 20)
        self.kc_lower = self.I(kc_lower, self.data.High, self.data.Low, self.data.Close, 20)
        
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
        self.trailing_stop = None
        self.peak_price = None

    def next(self):
        price = self.data.Close[-1]
        
        if not self.position:
            # Moon Dev Entry Logic 🌙🚀
            bb_squeeze = self.bb_width[-1] < self.bb_width_ma[-1]
            
            if bb_squeeze:
                # Long Entry - Moon Phase
                if price > self.kc_upper[-1]:
                    risk_amount = self.equity * self.risk_per_trade
                    atr_value = self.atr[-1]
                    units = int(round(risk_amount / (2 * atr_value)))
                    if units > 0:
                        self.buy(size=units)
                        self.peak_price = self.data.High[-1]
                        print(f"