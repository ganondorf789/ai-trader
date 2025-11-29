```python
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import talib

# Moon Dev Data Preparation 🌙
def prepare_data(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = prepare_data(data_path)

class VolatilityZipper(Strategy):
    risk_pct = 0.02
    stop_loss_pct = 0.01
    rr_ratio = 2
    consecutive_losses = 0
    in_cool_down = False

    def init(self):
        # Trend Filter 🌙
        self.sma20 = self.I(talib.SMA, self.data.Close, 20)
        
        # Volatility Engine ✨
        self.dispersion = self.I(talib.STDDEV, self.data.Close, 4)
        self.disp_ma = self.I(talib.SMA, self.dispersion, 20)
        self.disp_std = self.I(talib.STDDEV, self.dispersion, 20)
        
        # Volume Accelerator 🚀
        self.vol_ma = self.I(talib.SMA, self.data.Volume, 20)
        self.vol_accel = []
        self.vol_z = []

    def next(self):
        if len(self.data) < 20 or self.in_cool_down:
            return

        # Current Values 🌕
        close = self.data.Close[-1]
        sma = self.sma20[-1]
        
        # Dispersion Z-Score Calculation ✨
        disp_z = (self.dispersion[-1] - self.disp_ma[-1]) / self.disp_std[-1] if self.disp_std[-1] != 0 else 0
        
        # Volume Acceleration Z-Score 🚀
        vol_roc = (self.data.Volume[-1] - self.vol_ma[-1])/self.vol_ma[-1] if self.vol_ma[-1] != 0 else 0
        self.vol_accel.append(vol_roc)
        if len(self.vol_accel) >= 20:
            vol_z_current = (vol_roc - np.mean(self.vol_accel[-20:]))/np.std(self.vol_accel[-20:]) if np.std(self.vol_accel[-20:]) != 0 else 0
        else:
            vol_z_current = 0
        self.vol_z.append(vol_z_current)

        # Moon Dev Entry Logic 🌙
        if not self.position:
            # Long Entry 🌈
            if disp_z > 2 and vol_z_current > 2 and close > sma:
                self.enter_trade('long')
            
            # Short Entry 🌑
            elif disp_z < -2 and vol_z_current > 2 and close < sma:
                self.enter_trade('short')

        # Moon Dev Exit Logic 🎯
        if self.position.is_long and disp_z <= 0:
            self.position.close()
            print(f"🎯 Moon Dev Normalization! Closing long at {close}")
        if self.position.is_short and disp_z >= 0:
            self.position.close()
            print(f"🎯 Moon Dev Normalization! Closing short at {close}")

    def enter_trade(self, direction):
        risk_amount = self.risk_pct * self.equity
        price = self.data.Close[-1]
        
        if direction == 'long':
            sl = price * (1 - self.stop_loss_pct)
            tp = price * (1 + self.stop_loss_pct * self.rr_ratio)
            size = int(round(risk_amount / (price - sl)))
            if size > 0:
                self.buy(size=size, sl=sl, tp=tp)
                print(f"🚀 Moon Dev LONG! {size} units @ {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
        
        elif direction == 'short':
            sl = price * (1 + self.stop_loss_pct)
            tp