Here's the fixed code with all backtesting.lib references removed and proper indicator implementations:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Load and prepare data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class SqueezeSurge(Strategy):
    def init(self):
        # Bollinger Bands and Width
        self.bb_upper = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)[0], self.data.Close)
        self.bb_lower = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)[2], self.data.Close)
        self.bb_mid = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)[1], self.data.Close)
        self.bb_width = self.I(lambda u,l: (u-l)/talib.SMA(self.data.Close, timeperiod=20), self.bb_upper, self.bb_lower)
        self.bb_width_low = self.I(lambda x: talib.MIN(x, timeperiod=20), self.bb_width)
        
        # Trend and Volatility
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        # Volume
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
        self.trailing_high = 0
        self.trailing_low = np.inf

    def next(self):
        if len(self.data) < 40:
            return
        
        print(f"🌙 Bar {len(self.data)-1} | Close: {self.data.Close[-1]:.2f} | ADX: {self.adx[-1]:.2f} | BBW: {self.bb_width[-1]:.4f}")
        
        if not self.position:
            # Common conditions
            squeeze = self.bb_width[-1] <= self.bb_width_low[-1]
            weak_trend = self.adx[-1] < 25
            vol_surge = self.data.Volume[-1] > 1.5 * self.vol_ma[-1]
            
            # Long entry
            if squeeze and weak_trend and vol_surge and self.data.Close[-1] > self.bb_upper[-1]:
                risk_amount = self.equity * 0.01
                atr_val = self.atr[-1] or 1
                size = int(round(risk_amount / (2 * atr_val)))
                if size > 0:
                    self.buy(size=size)
                    self.trailing_high = self.data.High[-1]
                    print(f"🚀🌙 LONG! Size: {size} | Entry: {self.data.Close[-1]:.2f}")
            
            # Short entry
            elif squeeze and weak_trend and vol_surge and self.data.Close[-1] < self.bb_lower[-1]:
                risk_amount = self.equity * 0.01
                atr_val = self.atr[-1] or 1
                size = int(round(risk_amount / (2 * atr_val)))
                if size > 0:
                    self.sell(size=size)
                    self.trailing_low = self.data.Low[-1]
                    print(f"📉🌙 SHORT! Size: {size} | Entry: {self.data.Close[-1]:.2f}")
        
        # Manage exits
        elif self.position.is_long:
            self.trailing_high = max(self.trailing_high, self.data.High[-1])
            stop_level = self.trailing_high - 2 * self.atr[-1]
            if self.data.Low[-1] < stop_level:
                self.position.close()
                print(f"🔴