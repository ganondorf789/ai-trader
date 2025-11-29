Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed debug prints added:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION FOR SQUEEZE SURGE STRATEGY 🚀✨

import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
import warnings
warnings.filterwarnings('ignore')

# ======================
# DATA PREPARATION
# ======================
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename columns to match Backtesting.py requirements
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert and set datetime index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# ======================
# STRATEGY IMPLEMENTATION
# ======================
class SqueezeSurge(Strategy):
    risk_per_trade = 0.01  # 1% of equity per trade 🌙
    
    def init(self):
        # 🌗 BOLLINGER BAND CALCULATIONS
        self.bb_upper = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0], self.data.Close)
        self.bb_middle = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1], self.data.Close)
        self.bb_lower = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2], self.data.Close)
        
        # 🌌 BB WIDTH CALCULATION
        self.bb_width = self.I(lambda u, l, m: (u - l)/m, self.bb_upper, self.bb_lower, self.bb_middle)
        self.bb_width_sma200 = self.I(talib.SMA, self.bb_width, timeperiod=200)
        
        # 📈 VOLUME SURGE INDICATOR
        self.volume_pct90 = self.I(ta.percentile, self.data.Volume, length=20, percentile=90)
        
        # 🌀 ATR FOR RISK MANAGEMENT
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
        # Trackers for position management
        self.entry_price = None
        self.highest_high = None

    def next(self):
        # 🌑 SKIP EARLY BARS WITHOUT INDICATOR DATA
        if len(self.data) < 220 or np.isnan(self.bb_width_sma200[-1]):
            return
            
        # 🚀 LONG ENTRY CONDITIONS
        if not self.position:
            squeeze_condition = self.bb_width[-1] < 0.1 * self.bb_width_sma200[-1]
            volume_surge = self.data.Volume[-1] >= self.volume_pct90[-1]
            price_breakout = self.data.Close[-1] > self.bb_upper[-1]
            
            if squeeze_condition and volume_surge and price_breakout:
                # 🌕 POSITION SIZING CALCULATION
                atr_value = self.atr[-1]
                if atr_value <= 0:
                    return
                    
                risk_amount = self.equity * self.risk_per_trade
                position_size = risk_amount / atr_value
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    self.entry_price = self.data.Close[-1]
                    self.highest_high = self.data.High[-1]
                    print(f"🌕 MOON DEV LONG ENTRY! 🌕 | Size: {position_size} | Entry: {self.entry_price:.2f} | ATR: {atr_value:.2f}")
        
        # 🛑 POSITION MANAGEMENT
        else:
            # 🌙 UPDATE TRAILING HIGH
            self.highest