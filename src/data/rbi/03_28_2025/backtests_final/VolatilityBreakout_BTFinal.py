Here's the fixed and complete backtest code with Moon Dev themed debug prints and proper position sizing:

```python
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

# Load and preprocess data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolatilityBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Choppiness Index components
        self.tr = self.I(talib.TRANGE, self.data.High, self.data.Low, self.data.Close)
        self.sum_tr = self.I(talib.SUM, self.tr, 14)
        self.max_high = self.I(talib.MAX, self.data.High, 14)
        self.min_low = self.I(talib.MIN, self.data.Low, 14)
        
        # VWMA components
        self.close_volume = self.I(lambda: self.data.Close * self.data.Volume)
        self.numerator = self.I(talib.SUM, self.close_volume, 20)
        self.denominator = self.I(talib.SUM, self.data.Volume, 20)
        
        # Swing levels for SL
        self.swing_high = self.I(talib.MAX, self.data.High, 5)
        self.swing_low = self.I(talib.MIN, self.data.Low, 5)
        
        self.prev_ci = None
        self.prev_vwma = None

    def next(self):
        if len(self.data) < 20 or len(self.data) < 14:
            return
            
        # Calculate Choppiness Index
        current_sum_tr = self.sum_tr[-1]
        current_max_high = self.max_high[-1]
        current_min_low = self.min_low[-1]
        denominator = current_max_high - current_min_low
        ci = 100 * (np.log10(current_sum_tr/denominator)/np.log10(14)) if denominator > 0 else 0
        
        # Calculate VWMA
        current_vwma = self.numerator[-1]/self.denominator[-1] if self.denominator[-1] != 0 else 0
        
        if self.prev_ci is not None and self.prev_vwma is not None:
            # Long entry condition
            if (self.prev_ci >= 61.8 and ci < 61.8) and (self.data.Close[-2] < self.prev_vwma and self.data.Close[-1] > current_vwma):
                if not self.position:
                    sl = self.swing_low[-1] * 0.995
                    risk_per_share = self.data.Open[-1] - sl
                    if risk_per_share > 0:
                        size = int(round((self.equity * self.risk_per_trade) / risk_per_share))
                        if size > 0:  # Ensure we don't trade zero or negative size
                            self.buy(size=size, sl=sl, tp=self.data.Open[-1] + 2*risk_per_share)
                            print(f"🌙✨🚀 MOON DEV LONG ENTRY | Size: {size} | Entry: {self.data.Open[-1]:.2f} | SL: {sl:.2f} | TP: {self.data.Open[-1] + 2*risk_per_share:.2f}")
            
            # Short entry condition
            if (self.prev_ci >= 61.8 and ci < 61.8) and (self.data.Close[-2] > self.prev_vwma and self.data.Close[-1] < current_vwma):
                if not self.position:
                    sl = self.swing_high[-1] * 1.005
                    risk_per_share = sl - self.data.Open[-1]
                    if risk_per_share > 0:
                        size = int(round((self.equity * self.risk_per_trade) / risk_per_share))
                        if size > 0:  # Ensure we don't trade zero or negative size
                            self.sell(size=size, sl=sl, tp=self.data.Open[-1] - 2*risk_per