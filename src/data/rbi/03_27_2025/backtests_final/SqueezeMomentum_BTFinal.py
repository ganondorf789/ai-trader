I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete, debugged version with Moon Dev themed improvements:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib

class SqueezeMomentum(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Bollinger Bands
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name='UPPER', which=0)
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name='LOWER', which=2)
        
        # Chande Momentum Oscillator
        self.cmo = self.I(talib.CMO, self.data.Close, timeperiod=14, name='CMO')
        self.cmo_signal = self.I(talib.SMA, self.cmo, timeperiod=9, name='CMO_SIGNAL')
        
        # Volume MA
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VOLUME_MA')
        
        # Swing High/Low for stop loss
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='SWING_HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='SWING_LOW')
        
        # Squeeze Detection
        self.bandwidth = self.I(lambda u, l: (u - l) / l * 100, self.bb_upper, self.bb_lower, name='BANDWIDTH')
        self.min_bandwidth = self.I(talib.MIN, self.bandwidth, timeperiod=20, name='MIN_BW')

    def next(self):
        if not self.position:
            # Long Entry Conditions
            squeeze = self.bandwidth[-1] <= self.min_bandwidth[-1]
            cmo_up = (self.cmo[-2] < self.cmo_signal[-2]) and (self.cmo[-1] > self.cmo_signal[-1])
            volume_ok = self.data.Volume[-1] > self.volume_ma[-1]
            price_break = self.data.Close[-1] > self.bb_upper[-1]
            
            if squeeze and cmo_up and volume_ok and price_break:
                sl_price = min(self.swing_low[-1], self.bb_lower[-1])
                risk = self.data.Close[-1] - sl_price
                if risk <= 0: 
                    print("🌙 MOON DEV WARNING: Invalid risk calculation for long entry! ✨")
                    return
                
                size = int(round((self.equity * self.risk_per_trade) / risk))
                if size > 0:
                    self.buy(size=size, sl=sl_price, tag='Long')
                    print(f"🌙 MOON DEV LONG ALERT! Entered {size} units at {self.data.Close[-1]} ✨")
                else:
                    print("🌙 MOON DEV WARNING: Position size too small for long entry! ✨")
            
            # Short Entry Conditions
            cmo_down = (self.cmo[-2] > self.cmo_signal[-2]) and (self.cmo[-1] < self.cmo_signal[-1])
            price_break = self.data.Close[-1] < self.bb_lower[-1]
            
            if squeeze and cmo_down and volume_ok and price_break:
                sl_price = max(self.swing_high[-1], self.bb_upper[-1])
                risk = sl_price - self.data.Close[-1]
                if risk <= 0: 
                    print("🌙 MOON DEV WARNING: Invalid risk calculation for short entry! ✨")
                    return
                
                size = int(round((self.equity * self.risk_per_trade) / risk))
                if size > 0:
                    self.sell(size=size, sl=sl_price, tag='Short')
                    print(f"🌙 MOON DEV SHORT ALERT! Entered {size} units at {self.data.Close[-1]} ✨")
                else:
                    print("🌙 MOON DEV WARNING: Position size too small for short entry! ✨")
        else:
            # Exit Conditions
            if self.position.is_long:
                if (self.cmo_signal[-