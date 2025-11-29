```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib

# Data preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VortexDivergence(Strategy):
    risk_pct = 0.01
    swing_period = 20
    cks_length = 10
    atr_length = 20
    multiplier = 3

    def init(self):
        # Core indicators
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume)
        self.vi_plus = self.I(lambda: talib.VORTEX(self.data.High, self.data.Low, self.data.Close, 14)[0])
        self.vi_minus = self.I(lambda: talib.VORTEX(self.data.High, self.data.Low, self.data.Close, 14)[1])
        
        # Chande Kroll Stop components
        self.hh = self.I(talib.MAX, self.data.High, self.cks_length)
        self.ll = self.I(talib.MIN, self.data.Low, self.cks_length)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_length)
        self.cks_long = self.I(lambda: self.hh - self.multiplier * self.atr)
        self.cks_short = self.I(lambda: self.ll + self.multiplier * self.atr)
        
        # Divergence detection
        self.price_lows = self.I(talib.MIN, self.data.Low, self.swing_period)
        self.obv_lows = self.I(talib.MIN, self.obv, self.swing_period)
        self.price_highs = self.I(talib.MAX, self.data.High, self.swing_period)
        self.obv_highs = self.I(talib.MAX, self.obv, self.swing_period)

    def next(self):
        # Exit conditions
        for trade in self.trades:
            if trade.is_long and (self.data.Close[-1] < self.cks_long[-1] or crossover(self.vi_minus, self.vi_plus)):
                trade.close()
                print(f"🌙✨💼 Closing LONG at {self.data.Close[-1]} (SL: {self.cks_long[-1]:.2f})")
            elif trade.is_short and (self.data.Close[-1] > self.cks_short[-1] or crossover(self.vi_plus, self.vi_minus)):
                trade.close()
                print(f"🌙✨💼 Closing SHORT at {self.data.Close[-1]} (SL: {self.cks_short[-1]:.2f})")

        if self.position:
            return

        # Entry logic
        price_bull_div = len(self.price_lows) > self.swing_period and \
                        self.price_lows[-1] < self.price_lows[-self.swing_period-1] and \
                        self.obv_lows[-1] > self.obv_lows[-self.swing_period-1]

        price_bear_div = len(self.price_highs) > self.swing_period and \
                        self.price_highs[-1] > self.price_highs[-self.swing_period-1] and \
                        self.obv_highs[-1] < self.obv_highs[-self.swing_period-1]

        # Long entry
        if price_bull_div and crossover(self.vi_plus, self.vi_minus) and self.data.Close[-1] > self.cks_long[-1]:
            risk = self.data.Close[-1] - self.cks_long[-1]
            if risk <= 0: return
            size = int(round((self.equity * self.risk_pct) / risk))
            self.buy(size=size, sl=self.cks_long[-1])
            print(f"🌙✨🚀 LONG Entry {size} units at {self.data.Close[-1]} (SL: {self.cks_long[-1]:.2f})")