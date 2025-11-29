Here's the fixed code with proper indicator packages and Moon Dev themed debug prints:

```python
import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
import talib

class VolSqueezeSurge(Strategy):
    def init(self):
        self.upper_band = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0]
        self.lower_band = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2]
        self.bb_width = self.I(lambda u, l: u - l, self.upper_band, self.lower_band)
        self.bb_width_pct = self.I(lambda x: x.rolling(100).quantile(0.2), self.bb_width)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        self.breakout_high = None
        self.breakout_low = None

    def next(self):
        if not self.position:
            if len(self.bb_width) < 100 or len(self.volume_sma) < 1:
                return
            
            current_bb_width = self.bb_width[-1]
            current_bb_pct = self.bb_width_pct[-1]
            current_volume = self.data.Volume[-1]
            volume_sma = self.volume_sma[-1]

            if current_bb_width < current_bb_pct and current_volume > 1.5 * volume_sma:
                if self.data.Close[-1] > self.upper_band[-1]:
                    self.breakout_high = self.data.High[-1]
                    self.breakout_low = self.data.Low[-1]
                    entry_price = self.data.Open[0]
                    sl_price = self.breakout_low
                    risk_per_share = entry_price - sl_price
                    if risk_per_share <= 0:
                        return
                    risk_amount = 0.01 * self.equity
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.buy(size=position_size, sl=sl_price, tp=entry_price + 2 * risk_per_share)
                        print(f"🌙✨ LONG ENTRY! Price: {entry_price:.2f}, Size: {position_size}, SL: {sl_price:.2f}, TP: {entry_price + 2*risk_per_share:.2f} 🚀")

                elif self.data.Close[-1] < self.lower_band[-1]:
                    self.breakout_high = self.data.High[-1]
                    self.breakout_low = self.data.Low[-1]
                    entry_price = self.data.Open[0]
                    sl_price = self.breakout_high
                    risk_per_share = sl_price - entry_price
                    if risk_per_share <= 0:
                        return
                    risk_amount = 0.01 * self.equity
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.sell(size=position_size, sl=sl_price, tp=entry_price - 2 * risk_per_share)
                        print(f"🌙✨ SHORT ENTRY! Price: {entry_price:.2f}, Size: {position_size}, SL: {sl_price:.2f}, TP: {entry_price - 2*risk_per_share:.2f} 🚀")
        else:
            if self.position.is_long:
                trailing_stop = (self.breakout_high + self.breakout_low) / 2
                if self.data.Low[0] < trailing_stop:
                    self.position.close()
                    print(f"🌙 EXIT LONG: 50% Retracement at {trailing_stop:.2f} ✨🛑")
            elif self.position.is_short:
                trailing_stop = (self.breakout_high + self.breakout_low) / 2
                if self.data.High[0] > trailing_stop:
                    self.position.close()
                    print(f"🌙 EXIT SHORT: 50% Retracement at {trailing_stop:.2f} ✨🛑")

data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns