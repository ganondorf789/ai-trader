```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
import pandas as pd
import talib
import numpy as np

# Data preparation
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
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolatilitySqueezeBreakout(Strategy):
    risk_pct = 0.01
    bb_period = 20
    bb_dev = 2
    vol_ma_period = 50
    bb_width_lookback = 60
    atr_period = 14
    
    def init(self):
        self.upper = self.I(lambda c: talib.BBANDS(c, self.bb_period, self.bb_dev, self.bb_dev)[0], self.data.Close)
        self.middle = self.I(lambda c: talib.BBANDS(c, self.bb_period, self.bb_dev, self.bb_dev)[1], self.data.Close)
        self.lower = self.I(lambda c: talib.BBANDS(c, self.bb_period, self.bb_dev, self.bb_dev)[2], self.data.Close)
        self.bb_width = self.I(lambda u, l, m: (u - l)/m, self.upper, self.lower, self.middle)
        self.min_bb_width = self.I(talib.MIN, self.bb_width, self.bb_width_lookback)
        self.vol_ma = self.I(talib.SMA, self.data.Volume, self.vol_ma_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        self.entry_width = None
        self.entry_price = None

    def next(self):
        if not self.position:
            if self.bb_width[-1] <= self.min_bb_width[-1]:
                if self.data.Volume[-1] >= 3 * self.vol_ma[-1]:
                    if self.data.Close[-1] > self.upper[-1]:
                        sl = self.lower[-1]
                        risk = self.data.Close[-1] - sl
                        if risk <= 0: return
                        size = int(round((self.risk_pct * self.equity) / risk))
                        self.buy(size=size, sl=sl)
                        self.entry_width = self.bb_width[-1]
                        print(f"🌙 LONG ENTRY @ {self.data.Close[-1]:.2f} | Vol: {self.data.Volume[-1]:.2f} (3x MA) | BBW: {self.bb_width[-1]:.4f} 🚀")
                        
                    elif self.data.Close[-1] < self.lower[-1]:
                        sl = self.upper[-1]
                        risk = sl - self.data.Close[-1]
                        if risk <= 0: return
                        size = int(round((self.risk_pct * self.equity) / risk))
                        self.sell(size=size, sl=sl)
                        self.entry_width = self.bb_width[-1]
                        print(f"🌙 SHORT ENTRY @ {self.data.Close[-1]:.2f} | Vol: {self.data.Volume[-1]:.2f} (3x MA) | BBW: {self.bb_width[-1]:.4f} 🚀")
        else:
            if self.position.is_long:
                if self.data.Close[-1] < self.upper[-1] or self.bb_width[-1] >= 3 * self.entry_width:
                    self.position.close()
                    print(f"✨ EXIT LONG @ {self.data.Close[-1]:.2f} | {'BB Close' if self.data.Close[-1] < self.upper[-1] else 'Width Expand'} 🌙")
                else:
                    trail = self.data.High[-1] - 2 * self.atr[-1]
                    self.position.sl = max(self.position.sl, trail)
                    
            elif self.position.is_short:
                if self.data.Close[-1] > self.lower[-1] or self.bb_width[-1] >=