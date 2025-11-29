```python
import pandas as pd
import talib
from backtesting import Strategy, Backtest
from backtesting.lib import crossover

class VolSqueezeBreakout(Strategy):
    def init(self):
        self.bb_upper = self.I(lambda close: talib.BBANDS(close, 20, 2, 2, 0)[0], self.data.Close, name='BB_Upper')
        self.bb_middle = self.I(lambda close: talib.BBANDS(close, 20, 2, 2, 0)[1], self.data.Close, name='BB_Middle')
        self.bb_lower = self.I(lambda close: talib.BBANDS(close, 20, 2, 2, 0)[2], self.data.Close, name='BB_Lower')
        self.bb_width = self.I(lambda u, l, m: (u - l)/m, self.bb_upper, self.bb_lower, self.bb_middle, name='BB_Width')
        self.bb_percentile = self.I(lambda x: x.rolling(100).quantile(0.1), self.bb_width, name='BB_Pct')
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20, name='Vol_MA')
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, 14, name='ADX')
        self.entry_bar = None

    def next(self):
        if self.position:
            current_bar = len(self.data)-1
            if current_bar - self.entry_bar >= 480:
                self.position.close()
                print(f"🌕 MOON DEV MAX DURATION EXIT! Bars: {current_bar-self.entry_bar}")
                return
            
            adx_cross = self.adx[-1] > 25 and (self.adx[-2] <=25 if len(self.adx)>1 else False)
            if adx_cross:
                self.position.close()
                print(f"✨ ADX CROSS 25! EXIT @ {self.data.Close[-1]}")
                return
            
            if self.position.is_long and self.data.Close[-1] < self.bb_upper[-1]:
                self.position.close()
                print(f"🌙 CLOSE INSIDE BB! LONG EXIT @ {self.data.Close[-1]}")
                return
            
            if self.position.is_short and self.data.Close[-1] > self.bb_lower[-1]:
                self.position.close()
                print(f"🌑 CLOSE INSIDE BB! SHORT EXIT @ {self.data.Close[-1]}")
                return
            
            if self.data.Volume[-1] < self.volume_ma[-1]:
                self.position.close()
                print(f"🚨 VOLUME DROP! EMERGENCY EXIT @ {self.data.Close[-1]}")
                return
        else:
            if len(self.bb_percentile) < 1 or pd.isna(self.bb_percentile[-1]):
                return
            
            bb_squeeze = self.bb_width[-1] < self.bb_percentile[-1]
            vol_spike = self.data.Volume[-1] > 1.5 * self.volume_ma[-1]
            
            if bb_squeeze and vol_spike:
                if self.data.Close[-1] > self.bb_upper[-1]:
                    risk = 0.01 * self.equity
                    sl = self.bb_lower[-1]
                    risk_per_share = self.data.Close[-1] - sl
                    if risk_per_share <= 0:
                        risk_per_share = 0.01
                    size = int(round(risk / risk_per_share))
                    if size > 0:
                        self.buy(size=size, sl=sl)
                        self.entry_bar = len(self.data)-1
                        print(f"🌙✨ LONG! Size: {size}, Entry: {self.data.Close[-1]}, SL: {sl}")
                
                elif self.data.Close[-1] < self.bb_lower[-1]:
                    risk = 0.01 * self.equity
                    sl = self.bb_upper[-1]
                    risk_per_share = sl - self.data.Close[-1]
                    if risk_per_share <= 0:
                        risk_per_share = 0.01
                    size = int(round(risk / risk_per_share))
                    if size > 0:
                        self.sell(size=size, sl=sl)
                        self.entry_bar = len(self.data)-1
                        print(f"🌑🔄 SHORT! Size: {size}, Entry: {self.data.Close[-1]