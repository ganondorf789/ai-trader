I'll debug the code while maintaining the original strategy logic. Here's the fixed version with Moon Dev themed debug prints:

```python
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np
from backtesting import Backtest, Strategy

class VortexMomentumFilter(Strategy):
    vortex_period = 14
    cmo_period = 20
    volume_ma_period = 20
    swing_period = 20
    risk_percent = 0.01

    def init(self):
        # Calculate Vortex Indicator using pandas_ta
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)
        
        vi = ta.vortex(high=high, low=low, close=close, length=self.vortex_period)
        self.vi_plus = self.I(lambda: vi[f'VORTICS_{self.vortex_period}p'].fillna(0).values, name='VI+')
        self.vi_minus = self.I(lambda: vi[f'VORTICS_{self.vortex_period}r'].fillna(0).values, name='VI-')
        
        # Chande Momentum Oscillator
        self.cmo = self.I(talib.CMO, self.data.Close, self.cmo_period, name='CMO')
        
        # Volume MA and Swing Levels
        self.volume_ma = self.I(talib.SMA, self.data.Volume, self.volume_ma_period, name='Volume_MA')
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period, name='Swing_High')
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period, name='Swing_Low')

    def next(self):
        print(f"🌙 {self.data.index[-1]} | Close: {self.data.Close[-1]:.2f} | VI+ {self.vi_plus[-1]:.2f} > VI- {self.vi_minus[-1]:.2f}? | CMO: {self.cmo[-1]:.2f}")

        if not self.position:
            # Long entry checks
            if ((self.vi_plus[-2] < self.vi_minus[-2] and self.vi_plus[-1] > self.vi_minus[-1]) and \
               (self.cmo[-1] > 50) and \
               (self.data.Volume[-1] < self.volume_ma[-1]):
                
                risk_amount = self.equity * self.risk_percent
                entry_price = self.data.Close[-1]
                stop_price = self.swing_low[-1]
                risk_per_share = entry_price - stop_price
                
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    if size > 0:
                        self.buy(size=size, sl=stop_price)
                        print(f"🚀🌕 LONG! Moon Rocket Ignition! Size: {size} @ {entry_price:.2f}")
            
            # Short entry checks
            elif ((self.vi_minus[-2] < self.vi_plus[-2] and self.vi_minus[-1] > self.vi_plus[-1]) and \
                 (self.cmo[-1] < -50) and \
                 (self.data.Volume[-1] < self.volume_ma[-1]):
                
                risk_amount = self.equity * self.risk_percent
                entry_price = self.data.Close[-1]
                stop_price = self.swing_high[-1]
                risk_per_share = stop_price - entry_price
                
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    if size > 0:
                        self.sell(size=size, sl=stop_price)
                        print(f"📉🌑 SHORT! Lunar Eclipse Trading! Size: {size} @ {entry_price:.2f}")
        
        else:
            # Exit conditions
            if self.position.is_long and (self.cmo[-2] > 30 and self.cmo[-1] < 30):
                self.position.close()
                print(f"🌙✨ LONG EXIT! Moon Base Profits Secured!")
            elif self.position.is_short and (self.cmo[-2] < -30 and self.cmo[-1] > -30):
                self.position.close()
                print(f"🌙✨ SHORT EXIT! Escape From Black Hole!")

# Data preprocessing would go here
print("🌌✨ Moon Dev Debug Complete! Vortex Momentum