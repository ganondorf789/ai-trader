Here's the fixed code with all necessary adjustments and Moon Dev themed debug prints:

```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

# Load and preprocess data
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

class VolatilitySynchron(Strategy):
    risk_pct = 0.01
    atr_multiple = 2
    time_exit_bars = 5
    
    def init(self):
        # Keltner Channel indicators
        self.ema20 = self.I(talib.EMA, self.data.Close, timeperiod=20)
        self.atr20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=20)
        self.kc_width = self.I(lambda: 4 * self.atr20)
        self.kc_width_min = self.I(talib.MIN, self.kc_width, timeperiod=20)
        
        # CMF calculation
        mfm = (2*self.data.Close - self.data.High - self.data.Low)/(self.data.High - self.data.Low + 1e-9)
        mfm = np.where(self.data.High != self.data.Low, mfm, 0)
        mfv = mfm * self.data.Volume
        self.sum_mfv = self.I(talib.SUM, mfv, timeperiod=20)
        self.sum_vol = self.I(talib.SUM, self.data.Volume, timeperiod=20)
        
        # CMF/price swing lows
        self.cmf_low = self.I(talib.MIN, self.sum_mfv/self.sum_vol, timeperiod=5)
        self.price_low = self.I(talib.MIN, self.data.Low, timeperiod=5)
        
        # Position sizing ATR
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        self.entry_bar = 0
        print("🌌 MOON DEV INIT COMPLETE ✨ | Indicators loaded successfully")

    def next(self):
        if len(self.data) < 20:
            print("🌑 MOON DEV WARNING | Waiting for sufficient data...")
            return
            
        # Current values
        close = self.data.Close[-1]
        upper_kc = self.ema20[-1] + 2*self.atr20[-1]
        lower_kc = self.ema20[-1] - 2*self.atr20[-1]
        vol_avg = self.I(talib.SMA, self.data.Volume, timeperiod=20)[-1]
        
        # Keltner squeeze condition
        squeeze = self.kc_width[-1] <= self.kc_width_min[-1]
        
        # CMF divergence check
        cmf_lows = self.cmf_low[-3:]
        price_lows = self.price_low[-3:]
        bull_div = (len(cmf_lows) >=2 and len(price_lows)>=2 and 
                   price_lows[-1] < price_lows[-2] and 
                   cmf_lows[-1] > cmf_lows[-2])
        
        # Long entry logic
        if not self.position and squeeze and bull_div and close > upper_kc and self.data.Volume[-1] > vol_avg:
            atr = self.atr14[-1]
            risk_amount = self.equity * self.risk_pct
            position_size = int(round(risk_amount / (atr * self.atr_multiple)))
            
            if position_size > 0:
                self.buy(size=position_size, sl=lower_kc, tp=close + 2*atr)
                self.entry_bar = len(self.data)
                print(f"🌙 MOON DEV LONG ENTRY 🚀 | Price: {close:.2f} | Size: {position_size} | SL: {lower_kc:.2f} | TP: {close+2*atr:.2f}")
                print(f"🌕 POS