Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations:

```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import pandas_ta as ta
import numpy as np

# Data preprocessing
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VolatilityReversal(Strategy):
    def init(self):
        # Momentum indicators
        self.roc_3 = self.I(talib.ROC, self.data.Close, 3, name='ROC_3')
        self.atr_20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 20, name='ATR_20')
        
        # EMV calculation using pandas_ta
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)
        emv = ta.emv(high, low, volume, length=14)
        self.emv = self.I(lambda: emv['EMV_14'].values, name='EMV')
        
        # Volume and price structure
        self.vol_ma = self.I(talib.SMA, self.data.Volume, 20, name='Volume_MA')
        self.swing_high = self.I(talib.MAX, self.data.High, 20, name='Swing_High')
        self.swing_low = self.I(talib.MIN, self.data.Low, 20, name='Swing_Low')
        
        # Pattern recognition
        self.hammer = self.I(talib.CDLHAMMER, self.data.Open, self.data.High, self.data.Low, self.data.Close, name='Hammer')
        self.engulfing = self.I(talib.CDLENGULFING, self.data.Open, self.data.High, self.data.Low, self.data.Close, name='Engulfing')
        
        print("🌙 Lunar indicators initialized! Ready for launch 🚀")

    def next(self):
        if self.position:
            return  # Maintain existing position

        # Current values
        close = self.data.Close[-1]
        atr_pct = (self.atr_20[-1]/close)*100
        roc = self.roc_3[-1]
        vol = self.data.Volume[-1]
        emv_slope = self.emv[-1] < self.emv[-2] if len(self.emv) > 1 else False

        # Long conditions
        if (roc > 2*atr_pct and emv_slope and 
            (self.hammer[-1] > 0 or self.engulfing[-1] > 0) and 
            vol > self.vol_ma[-1]):
            
            sl = self.swing_low[-1] - self.atr_20[-1]
            risk = close - sl
            position_size = int(round((self.equity*0.01)/risk))
            
            if position_size > 0:
                height = self.swing_high[-1] - self.swing_low[-1]
                targets = [self.swing_high[-1] + height*1.618,
                          self.swing_high[-1] + height*2.0]
                tp = min(t for t in targets if t > close)
                self.buy(size=position_size, sl=sl, tp=tp)
                print(f"🌙🚀 BULLISH REVERSAL! Long {position_size} @ {close}")
                print(f"   SL: {sl:.2f} | TP: {tp:.2f} 🌕")

        # Short conditions
        elif (roc < -2*atr_pct and emv_slope and 
              (self.engulfing[-1] < 0) and 
              vol > self.vol_ma[-1]):
            
            sl = self.swing_high[-1] + self.atr_20[-1]
            risk = sl - close
            position_size = int(round((self.equity*0.01)/risk))
            
            if position_size > 0: