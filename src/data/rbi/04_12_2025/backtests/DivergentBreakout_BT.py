```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
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

class DivergentBreakout(Strategy):
    adx_entry_threshold = 25
    adx_exit_threshold = 30
    obv_ma_period = 20
    swing_period = 5
    risk_pct = 0.01

    def init(self):
        # Core indicators
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume.astype(float))
        self.obv_sma = self.I(talib.SMA, self.obv, self.obv_ma_period)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, 14)
        
        # Divergence detection
        self.swing_low_price = self.I(talib.MIN, self.data.Low, self.swing_period)
        self.swing_low_obv = self.I(talib.MIN, self.obv, self.swing_period)
        self.swing_high_price = self.I(talib.MAX, self.data.High, self.swing_period)
        self.swing_high_obv = self.I(talib.MAX, self.obv, self.swing_period)

    def next(self):
        if len(self.data) < self.swing_period + 2:
            return

        # Current values
        current_adx = self.adx[-1]
        current_close = self.data.Close[-1]

        # Divergence checks
        bull_div = (self.swing_low_price[-1] < self.swing_low_price[-2]) and \
                  (self.swing_low_obv[-1] > self.swing_low_obv[-2])
        bear_div = (self.swing_high_price[-1] > self.swing_high_price[-2]) and \
                   (self.swing_high_obv[-1] < self.swing_high_obv[-2])

        # OBV crossover checks
        obv_above = crossover(self.obv[-2:], self.obv_sma[-2:])[-1]
        obv_below = crossunder(self.obv[-2:], self.obv_sma[-2:])[-1]

        # Entry logic
        if not self.position:
            # Long entry
            if bull_div and current_adx < self.adx_entry_threshold and obv_above:
                sl = self.swing_low_price[-1]
                risk_per_share = current_close - sl
                if risk_per_share > 0:
                    size = int(round((self.equity * self.risk_pct) / risk_per_share))
                    if size > 0:
                        tp = current_close + 2 * risk_per_share
                        self.buy(size=size, sl=sl, tp=tp)
                        print(f"🚀🌙 BULLISH BREAKOUT! Entry: {current_close:.2f}, Size: {size}, SL: {sl:.2f}, TP: {tp:.2f}")

            # Short entry
            elif bear_div and current_adx < self.adx_entry_threshold and obv_below:
                sl = self.swing_high_price[-1]
                risk_per_share = sl - current_close
                if risk_per_share > 0:
                    size = int(round((self.equity * self.risk_pct) / risk_per_share))
                    if size > 0:
                        tp = current_close - 2 * risk_per_share
                        self.sell(size=size, sl=sl, tp=tp)
                        print(f"🌑🌙 BEARISH BREAKOUT! Entry: {current_close:.2f}, Size: {size}, SL: {sl:.2f}, TP: {tp:.2f}")

        # Exit logic
        if self.position and current_adx > self.adx_ex