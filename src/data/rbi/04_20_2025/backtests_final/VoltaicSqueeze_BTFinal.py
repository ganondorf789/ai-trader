```python
from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta

# Data preprocessing
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
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
data = data.set_index('datetime')

class VoltaicSqueeze(Strategy):
    def init(self):
        # Bollinger Bands
        self.bb_upper = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0], self.data.Close)
        self.bb_middle = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1], self.data.Close)
        self.bb_lower = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2], self.data.Close)
        
        # BB Width Calculation
        self.bb_width = self.I(lambda u, l, m: (u - l) / m, self.bb_upper, self.bb_lower, self.bb_middle)
        
        # BB Width Percentile using rolling rank
        self.bb_rank = self.I(talib.LINEARREG_ANGLE, self.bb_width, timeperiod=100)
        self.bb_width_percentile = self.I(lambda x: (x - 1) * (100/99), self.bb_rank)
        
        # Keltner Channels
        self.ema20 = self.I(talib.EMA, self.data.Close, 20)
        self.atr20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 20)
        self.keltner_upper = self.I(lambda e, a: e + 2*a, self.ema20, self.atr20)
        self.keltner_lower = self.I(lambda e, a: e - 2*a, self.ema20, self.atr20)
        
        # Volume SMA
        self.volume_sma20 = self.I(talib.SMA, self.data.Volume, 20)
        
        # ATR for stops
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
    def next(self):
        if not self.position:
            # Long entry conditions
            if (self.bb_width_percentile[-1] < 20 and
                self.keltner_upper[-1] < self.bb_upper[-1] and
                self.keltner_lower[-1] > self.bb_lower[-1] and
                self.data.Close[-1] > self.keltner_upper[-1] and
                self.data.Volume[-1] > self.volume_sma20[-1] and
                self.bb_width[-1] > self.bb_width[-2]):
                
                risk_amount = self.equity * 0.01
                atr = self.atr14[-1]
                entry_price = self.data.Close[-1]
                sl = entry_price - atr
                position_size = int(round(risk_amount / atr))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=sl, tp=entry_price + 2*atr)
                    print(f"🌙 MOON DEV LONG ENTRY 🚀 | Price: {entry_price:.2f} | Size: {position_size} | SL: {sl:.2f} | TP: {entry_price + 2*atr:.2f}")
            
            # Short entry conditions
            elif (self.bb_width_percentile[-1] < 20 and
                  self.keltner_upper[-1] < self.bb_upper[-1] and
                  self.keltner_lower[-1] > self.bb_lower[-1] and
                  self.data.Close[-1] < self.keltner_lower[-1] and
                  self.data.Volume[-1