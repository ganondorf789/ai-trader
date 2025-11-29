Here's the complete implementation of the VolCompressSpike strategy for backtesting.py:

```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and format data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolCompressSpike(Strategy):
    bb_period = 20
    bb_dev = 2
    vol_lookback = 30 * 24 * 4  # 30 days in 15m intervals
    bb_perc_lookback = 6 * 30 * 24 * 4  # 6 months
    risk_pct = 0.01
    max_losses = 3

    def init(self):
        # Bollinger Bands components
        def bb_upper(close):
            upper, _, _ = talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)
            return upper
        self.upper = self.I(bb_upper, self.data.Close)
        
        def bb_lower(close):
            _, _, lower = talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)
            return lower
        self.lower = self.I(bb_lower, self.data.Close)
        
        self.middle = self.I(talib.SMA, self.data.Close, self.bb_period)
        
        # Volatility compression indicator
        bb_width = (self.upper - self.lower) / self.middle
        self.bb_perc = self.I(ta.percentile, bb_width, self.bb_perc_lookback)
        
        # Volume indicators
        self.vol_ma = self.I(talib.SMA, self.data.Volume, self.vol_lookback)
        
        # Trade tracking
        self.consecutive_losses = 0
        self.low_vol_count = 0

    def next(self):
        if len(self.data) < self.bb_perc_lookback:
            return

        # Entry conditions
        if not self.position and self.consecutive_losses < self.max_losses:
            vol_spike = self.data.Volume[-1] > 3 * self.vol_ma[-1]
            compression = self.bb_perc[-1] <= 6
            
            if vol_spike and compression:
                close = self.data.Close[-1]
                if close > self.upper[-1]:  # Long setup
                    sl = (self.data.High[-1] + self.data.Low[-1]) / 2
                    risk = close - sl
                    size = int(round((self.risk_pct * self.equity) / risk))
                    if size > 0:
                        tp = self.middle[-1] + 1.5 * (self.upper[-1] - self.middle[-1])/self.bb_dev
                        self.buy(size=size, sl=sl, tp=tp)
                        print(f"🌕 MOON ROCKET LAUNCH! LONG {size} @ {close}")
                
                elif close < self.lower[-1]:  # Short setup
                    sl = (self.data.High[-1] + self.data.Low[-1]) / 2
                    risk = sl - close
                    size = int(round((self.risk_pct * self.equity) / risk))
                    if size > 0:
                        tp = self.middle[-1] - 1.5 * (self.middle[-1] - self.lower[-1])/self.bb_dev
                        self.sell(size=size, sl=sl, tp=tp)
                        print(f"🌑 DARK SIDE ENTRY! SHORT {size} @ {close}")

        # Volume-based exit
        if self.position:
            if self.data.Volume[-1] < 0.5 * self.vol_ma[-1]:
                self.low_vol_count += 1
                if self.low_vol_count >= 3:
                    self.position.close()
                    print(f"🌧️ VOLUME DROUGHT