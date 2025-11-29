Here's the fully debugged and Moon Dev themed backtest code with all technical issues fixed while preserving the original strategy logic:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Moon Dev Data Preparation Ritual 🌙✨
def load_data(path):
    data = pd.read_csv(path, parse_dates=['datetime'])
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    data.set_index('datetime', inplace=True)
    print("🌙 MOON DEV: Data successfully aligned with cosmic frequencies! 🌌")
    return data

class SqueezeBreakout(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌕
    
    def init(self):
        # Moon Dev Indicator Magic 🌙📈
        self.bb_upper = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)[0], 
                              self.data.Close, name='BB_UPPER')
        self.bb_lower = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)[2], 
                              self.data.Close, name='BB_LOWER')
        self.bb_width = self.I(lambda u, l: u - l, self.bb_upper, self.bb_lower, name='BB_WIDTH')
        self.bb_width_min = self.I(lambda x: talib.MIN(x, timeperiod=20), self.bb_width, name='BB_WIDTH_MIN')
        
        self.macd_line = self.I(lambda c: talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)[0], 
                               self.data.Close, name='MACD')
        self.macd_signal = self.I(lambda c: talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)[1], 
                                 self.data.Close, name='MACD_SIGNAL')
        
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                         timeperiod=14, name='ATR')
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VOLUME_SMA')
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=50, name='EMA50')
        
        print("🌙 MOON DEV: Indicators charged with lunar energy! 🚀✨")

    def next(self):
        if len(self.data) < 50:  # Wait for full cosmic alignment 🌌
            return

        # Lunar Signal Detection 🌙🔭
        squeeze = self.bb_width[-1] <= self.bb_width_min[-1]
        price_breakout = self.data.Close[-1] > self.bb_upper[-1]
        macd_bullish = (self.macd_line[-2] < self.macd_signal[-2] and 
                        self.macd_line[-1] > self.macd_signal[-1])
        volume_ok = self.data.Volume[-1] > self.volume_sma[-1]
        ema_slope = self.ema50[-1] > self.ema50[-2]

        # Cosmic Entry Portal 🌠🚪
        if not self.position and all([squeeze, price_breakout, macd_bullish, volume_ok, ema_slope]):
            atr_value = self.atr[-1]
            risk_amount = self.equity * self.risk_percent
            stop_loss = self.data.Close[-1] - atr_value * 1
            position_size = risk_amount / (self.data.Close[-1] - stop_loss)
            position_size = int(round(position_size))  # Moon Dev precision rounding 🌕
            
            if position_size > 0:
                self.buy(size=position_size, sl=stop_loss)
                self.trailing_atr = atr_value * 3  # 3x ATR trailing
                self.highest_high = self.data.High[-1]
                print(f"🌙🚀 MOON BLASTOFF! Long {position_size} units @ {self.data.Close[-1]:.2f}")
                print(f"   🌑 Initial