Here's the complete fixed code with all backtesting.lib imports removed and proper Moon Dev themed debug prints: 🌙✨

```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
import numpy as np

# Data preprocessing
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VoltaicFisher(Strategy):
    def init(self):
        # Core indicators 🌙
        self.sma = self.I(talib.SMA, self.data.Close, timeperiod=20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=20)
        
        # Volatility bands ✨
        self.upper_band = self.I(lambda: self.sma.array + 2 * self.atr.array)
        self.lower_band = self.I(lambda: self.sma.array - 2 * self.atr.array)
        
        # Fisher Transform with pandas_ta 🐟
        fisher, signal = ta.fisher(high=self.data.High, low=self.data.Low, length=14, signal=3)
        self.fisher = self.I(lambda: fisher)
        self.signal = self.I(lambda: signal)
        
        # Trend following tools 🛑
        self.sar = self.I(talib.SAR, self.data.High, self.data.Low, acceleration=0.02, maximum=0.2)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20)

    def next(self):
        # Position management system 🌗
        if self.position:
            if self.position.is_long:
                # SAR exit condition 🛑
                if self.sar[-1] > self.data.Close[-1]:
                    self.position.close()
                    print(f"🌙✨ EXIT LONG | SAR flipped above at {self.sar[-1]:.2f}")
                # Fisher reversal exit 🎣
                elif (self.signal[-2] < self.fisher[-2] and self.signal[-1] > self.fisher[-1]):
                    self.position.close()
                    print("🌙 EXIT LONG | Fisher momentum lost")
            
            elif self.position.is_short:
                # SAR exit condition 🛑
                if self.sar[-1] < self.data.Close[-1]:
                    self.position.close()
                    print(f"🌙✨ EXIT SHORT | SAR flipped below at {self.sar[-1]:.2f}")
                # Fisher reversal exit 🎣
                elif (self.fisher[-2] < self.signal[-2] and self.fisher[-1] > self.signal[-1]):
                    self.position.close()
                    print("🌙 EXIT SHORT | Fisher momentum lost")
        
        else:
            # Long entry constellation 🌌
            if (self.data.Close[-2] > self.upper_band[-2] and
                self.fisher[-2] > self.signal[-2] and
                self.fisher[-3] <= self.signal[-3]):
                
                risk_price = self.swing_low[-2]
                entry_price = self.data.Open[-1]
                risk_per_share = entry_price - risk_price
                
                if risk_per_share > 0:
                    position_size = int(round((self.equity * 0.01) / risk_per_share))
                    if position_size:
                        self.buy(size=position_size, sl=risk_price)
                        print(f"🚀🌙 LONG ENTRY | Size: {position_size} @ {entry_price:.2f} | SL: {risk_price:.2f}")
            
            # Short entry constellation 🌌
            if (self.data.Close[-2] < self.lower_band[-2] and
                self.fisher[-2] < self.signal[-2] and
                self.fisher[-3] >= self.signal[-3]):
                
                risk_price = self.swing_high[-2]
                entry_price = self.data.Open[-1]
                risk_per_share = risk_price - entry_price