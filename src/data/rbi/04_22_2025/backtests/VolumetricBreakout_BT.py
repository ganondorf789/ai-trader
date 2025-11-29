```python
import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
import talib

class VolumetricBreakoutStrategy(Strategy):
    def init(self):
        self.ema_period = 3 * 96  # 3 days in 15m periods
        self.high_period = 20 * 96  # 20 days
        self.low_period = 20 * 96
        self.volume_period = 30 * 96  # 30 days
        
        self.max_high = self.I(talib.MAX, self.data.High, timeperiod=self.high_period)
        self.min_low = self.I(talib.MIN, self.data.Low, timeperiod=self.low_period)
        self.ema3 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        self.volume_sma30 = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_period)
        
    def next(self):
        if self.position:
            if self.position.is_long:
                if self.data.Close[-1] < self.ema3[-1]:
                    self.sell()
                    print(f"🌙✨ Moon Dev Trailing Stop Exit Long at {self.data.Close[-1]:.2f}")
                elif len(self.data) - self.position.entry_bar >= 480:
                    self.sell()
                    print(f"🕒🌙 Moon Dev Time Exit Long after 5 days at {self.data.Close[-1]:.2f}")
            elif self.position.is_short:
                if self.data.Close[-1] > self.ema3[-1]:
                    self.cover()
                    print(f"🌙✨ Moon Dev Trailing Stop Exit Short at {self.data.Close[-1]:.2f}")
                elif len(self.data) - self.position.entry_bar >= 480:
                    self.cover()
                    print(f"🕒🌙 Moon Dev Time Exit Short after 5 days at {self.data.Close[-1]:.2f}")
        else:
            if len(self.data) < self.high_period:
                return
            
            current_close = self.data.Close[-1]
            max_high = self.max_high[-1]
            min_low = self.min_low[-1]
            ema3 = self.ema3[-1]
            volume = self.data.Volume[-1]
            volume_sma = self.volume_sma30[-1]
            
            if np.isnan([max_high, min_low, ema3, volume_sma]).any():
                return
            
            volume_condition = volume > 1.5 * volume_sma
            
            if current_close > max_high and current_close > ema3 and volume_condition:
                sl = min_low
                risk = current_close - sl
                if risk <= 0:
                    return
                risk_amount = self.equity * 0.01
                size = int(round(risk_amount / risk))
                if size > 0:
                    tp = current_close + 2 * risk
                    self.buy(size=size, sl=sl, tp=tp)
                    print(f"🚀🌙 Moon Dev LONG Entry! Price: {current_close:.2f}, Size: {size}, SL: {sl:.2f}, TP: {tp:.2f}")
            
            elif current_close < min_low and current_close < ema3 and volume_condition:
                sl = max_high
                risk = sl - current_close
                if risk <= 0:
                    return
                risk_amount = self.equity * 0.01
                size = int(round(risk_amount / risk))
                if size > 0:
                    tp = current_close - 2 * risk
                    self.sell(size=size, sl=sl, tp=tp)
                    print(f"🌑🌙 Moon Dev SHORT Entry! Price: {current_close:.2f}, Size: {size}, SL: {sl:.2f}, TP: {tp:.2f}")

data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
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

bt =