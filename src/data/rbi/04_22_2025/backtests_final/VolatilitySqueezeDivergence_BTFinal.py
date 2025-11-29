```python
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

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

class VolatilitySqueezeDivergence(Strategy):
    risk_per_trade = 0.01
    consecutive_losses = 0
    atr_period = 14
    kc_period = 20
    kc_multiplier = 2.0
    
    def init(self):
        # Calculate indicators
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.kc_period)
        self.atr_kc = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.kc_period)
        self.upper_kc = self.I(lambda: self.ema + self.atr_kc * self.kc_multiplier)
        self.lower_kc = self.I(lambda: self.ema - self.atr_kc * self.kc_multiplier)
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        
        # Track squeeze conditions
        self.squeeze_count = self.I(lambda: np.zeros(len(self.data)), plot=False)
        self.obv_highs = self.I(talib.MAX, self.obv, timeperiod=5)
        self.obv_lows = self.I(talib.MIN, self.obv, timeperiod=5)
        
    def next(self):
        if len(self.data) < 50 or self.position:
            return
            
        # Update squeeze count
        if (self.data.High[-1] < self.upper_kc[-1] and self.data.Low[-1] > self.lower_kc[-1]):
            self.squeeze_count[-1] = self.squeeze_count[-2] + 1
        else:
            self.squeeze_count[-1] = 0
            
        # Detect OBV divergence
        price_high = self.data.High[-5:].argmax()
        obv_high = self.obv[-5:].argmax()
        price_low = self.data.Low[-5:].argmin()
        obv_low = self.obv[-5:].argmin()
        
        long_div = (price_high < obv_high) and (self.obv[-1] > self.obv_highs[-1])
        short_div = (price_low > obv_low) and (self.obv[-1] < self.obv_lows[-1])
        
        # Entry conditions
        if self.squeeze_count[-1] >= 5:
            if (self.data.Close[-1] > self.upper_kc[-1] and
                self.data.Volume[-1] > self.volume_sma[-1] * 1.2 and
                long_div):
                
                sl_price = min(self.data.Low[-1], self.lower_kc[-1]) - self.atr[-1]
                risk = self.data.Close[-1] - sl_price
                position_size = int(round((self.risk_per_trade * self.equity) / risk))
                
                print(f"🌙 MOON DEV LONG SIGNAL ✨ | Price: {self.data.Close[-1]:.2f} | Size: {position_size} | SL: {sl_price:.2f} 🚀")
                self.buy(size=position_size, sl=sl_price, tp=self.data.Close[-1] + 2*self.atr[-1])
                
            elif (self.data.Close[-1] < self.lower_kc[-1] and
                  self.data.Volume[-1] > self.volume_sma[-1] * 1.2 and
                  short_div):
                  
                sl_price = max(self.data.High[-1], self.upper_kc[-