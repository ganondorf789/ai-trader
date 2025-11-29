```python
from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import talib

class VolcanicBreakout(Strategy):
    donchian_period = 20
    vma_period = 50
    vma_lookback = 200
    bb_period = 20
    bb_dev = 2
    risk_pct = 0.01

    def init(self):
        self.donchian_upper = self.I(talib.MAX, self.data.High, timeperiod=self.donchian_period)
        self.donchian_lower = self.I(talib.MIN, self.data.Low, timeperiod=self.donchian_period)
        self.vma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vma_period)
        
        def bbands(close):
            upper, middle, lower = talib.BBANDS(close, timeperiod=self.bb_period, 
                                               nbdevup=self.bb_dev, nbdevdn=self.bb_dev)
            return upper, middle, lower
        self.bb_upper, _, self.bb_lower = self.I(bbands, self.data.Close)

    def next(self):
        if len(self.data) < max(self.donchian_period, self.vma_lookback, self.bb_period):
            return

        current_close = self.data.Close[-1]
        donchian_upper = self.donchian_upper[-1]
        donchian_lower = self.donchian_lower[-1]
        donchian_mid = (donchian_upper + donchian_lower) / 2
        current_vma = self.vma[-1]

        vma_window = self.vma[-self.vma_lookback:]
        if len(vma_window) < self.vma_lookback:
            return
        vma_90 = np.percentile(vma_window, 90)

        bb_width = self.bb_upper[-1] - self.bb_lower[-1]
        bb_width_window = [self.bb_upper[i] - self.bb_lower[i] 
                          for i in range(-self.vma_lookback, 0)]
        bb_20 = np.percentile(bb_width_window, 20) if bb_width_window else 0

        if not self.position:
            if current_close > donchian_upper and current_vma > vma_90:
                risk_amount = self.equity * self.risk_pct
                risk_per_share = current_close - donchian_lower
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    self.buy(size=size, sl=donchian_lower)
                    print(f"🌋 LONG ERUPTION! Entry: {current_close:.2f} | Size: {size} ✨")

            elif current_close < donchian_lower and current_vma > vma_90:
                risk_amount = self.equity * self.risk_pct
                risk_per_share = donchian_upper - current_close
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    self.sell(size=size, sl=donchian_upper)
                    print(f"🌑 SHORT ERUPTION! Entry: {current_close:.2f} | Size: {size} ✨")

        else:
            if self.position.is_long:
                if bb_width <= bb_20 or current_close < donchian_mid:
                    self.position.close()
                    print(f"🚨 LONG EXIT! Price: {current_close:.2f} | Reason: {'Squeeze' if bb_width <= bb_20 else 'Mid breach'} 🌙")
            elif self.position.is_short:
                if bb_width <= bb_20 or current_close > donchian_mid:
                    self.position.close()
                    print(f"🚨 SHORT EXIT! Price: {current_close:.2f} | Reason: {'Squeeze' if bb_width <= bb_20 else 'Mid breach'} 🌙")

data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}, inplace=True)
data['datetime'] = pd.to_datetime(data