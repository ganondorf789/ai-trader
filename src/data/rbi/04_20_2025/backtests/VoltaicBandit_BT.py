```python
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib

# Data preparation
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
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VoltaicBandit(Strategy):
    ema_period = 20
    atr_period = 14
    rsi_period = 14
    volume_ma_period = 20
    risk_percent = 0.01
    
    def init(self):
        self.ema = self.I(talib.EMA, self.data.Close, self.ema_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period)
        self.volume_ma = self.I(talib.SMA, self.data.Volume, self.volume_ma_period)
        
    def next(self):
        current_time = self.data.index[-1].time()
        if (current_time >= pd.Timestamp('00:00').time()) and (current_time <= pd.Timestamp('00:30').time()):
            return
        if current_time >= pd.Timestamp('23:30').time():
            return
            
        if len(self.ema) < 2 or len(self.atr) < 2 or len(self.rsi) < 2:
            return
            
        upper_band = self.ema[-1] + 2 * self.atr[-1]
        lower_band = self.ema[-1] - 2 * self.atr[-1]
        prev_upper_band = self.ema[-2] + 2 * self.atr[-2]
        prev_lower_band = self.ema[-2] - 2 * self.atr[-2]
        
        if not self.position:
            if self.data.Close[-1] > upper_band and self.data.Close[-2] <= prev_upper_band:
                if self.data.Volume[-1] > self.volume_ma[-1]:
                    sl_price = self.ema[-1] - self.atr[-1]
                    risk_per_share = self.data.Close[-1] - sl_price
                    if risk_per_share <= 0:
                        return
                    position_size = int(round((self.risk_percent * self.equity) / risk_per_share))
                    if position_size > 0:
                        self.buy(size=position_size, sl=sl_price)
                        print(f"🌙✨ MOON DEV LONG ENTRY! {self.data.index[-1]} | Size: {position_size} | SL: {sl_price:.2f} 🚀")
                        
            elif self.data.Close[-1] < lower_band and self.data.Close[-2] >= prev_lower_band:
                if self.data.Volume[-1] > self.volume_ma[-1]:
                    sl_price = self.ema[-1] + self.atr[-1]
                    risk_per_share = sl_price - self.data.Close[-1]
                    if risk_per_share <= 0:
                        return
                    position_size = int(round((self.risk_percent * self.equity) / risk_per_share))
                    if position_size > 0:
                        self.sell(size=position_size, sl=sl_price)
                        print(f"🌙✨ MOON DEV SHORT ENTRY! {self.data.index[-1]} | Size: {position_size} | SL: {sl_price:.2f} 🚀")
        else:
            if self.position.is_long:
                if self.rsi[-1] < 50 and self.rsi[-2] >= 50:
                    self.position.close()
                    print(f"🌙🛑 MOON DEV LONG EXIT! RSI Cross 50 | Price: {self.data.Close[-1]} | {self.data.index[-1]}")
            elif self.position.is_short:
                if self.rsi[-1] > 50 and self.rsi[-2] <= 50:
                    self.position.close()
                    print(f"🌙🛑 MOON