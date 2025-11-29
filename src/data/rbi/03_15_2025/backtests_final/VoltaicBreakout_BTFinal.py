Here's the complete fixed implementation of the VoltaicBreakout strategy with all necessary corrections and Moon Dev themed debug prints:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION ✨

import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# DATA PREPARATION 🌌
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and format columns 🧹
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert and index datetime ⏳
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# Add end-of-day marker 🌙
data['day'] = data.index.date
data['is_last'] = data['day'] != data['day'].shift(-1)
data['is_last'] = data['is_last'].fillna(False)
data = data.drop(columns=['day'])

class VoltaicBreakout(Strategy):
    # STRATEGY PARAMETERS 🔧
    atr_period = 14
    atr_sma_period = 50
    atr_multiplier = 1.5
    risk_pct = 0.01  # 1% risk per trade 🌕
    
    def init(self):
        # VOLATILITY INDICATORS 🌋
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.atr_sma = self.I(talib.SMA, self.atr, self.atr_sma_period)
        
        # PREVIOUS DAY LEVELS 📅
        day_high = self.I(talib.MAX, self.data.High, 96)  # 96 periods = 24h
        self.prev_day_high = self.I(lambda x: np.concatenate([np.full(96, np.nan), x[:-96]]), day_high)
        day_low = self.I(talib.MIN, self.data.Low, 96)
        self.prev_day_low = self.I(lambda x: np.concatenate([np.full(96, np.nan), x[:-96]]), day_low)
        
        # EOD MARKER 🕒
        self.is_last = self.I(lambda x: x, self.data.is_last)

    def next(self):
        # WAIT FOR SUFFICIENT DATA 🕰️
        if len(self.data) < 200 or len(self.atr) < self.atr_sma_period:
            return

        # GET INDICATOR VALUES 📊
        prev_high = self.prev_day_high[-1]
        prev_low = self.prev_day_low[-1]
        atr = self.atr[-1]
        atr_sma = self.atr_sma[-1]

        # LONG ENTRY CONDITIONS 🚀
        if (self.data.Close[-2] < prev_high and self.data.Close[-1] > prev_high) and atr > atr_sma * self.atr_multiplier:
            self.calculate_position(long=True, entry=self.data.Open[-1], 
                                 sl=prev_low, atr=atr)
            print(f"🌙✨ Moon Dev Long Signal! Entry: {self.data.Open[-1]}")

        # SHORT ENTRY CONDITIONS 🌑
        elif (self.data.Close[-2] > prev_low and self.data.Close[-1] < prev_low) and atr > atr_sma * self.atr_multiplier:
            self.calculate_position(long=False, entry=self.data.Open[-1],
                                 sl=prev_high, atr=atr)
            print(f"🌙✨ Moon Dev Short Signal! Entry: {self.data.Open[-1]}")

        # END-OF-DAY EXIT 🌅
        if self.is_last[-1] and self.position:
            self.position.close()
            print(f"🌙✨ Moon Dev Time Exit! Closing at {self.data.Close[-1]}")

    def calculate_position(self, long, entry, sl, atr):
        # RISK CALCULATION ⚖️
        risk_per_share = abs(entry - sl)
        if risk_per_share == 0:
            return