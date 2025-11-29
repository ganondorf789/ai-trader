Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations:

```python
# moon_dev_backtest.py 🌙🚀
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class MomentumSqueeze(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Clean data columns
        self.data.columns = [col.strip().lower() for col in self.data.columns]
        
        # Calculate indicators with proper TA-Lib wrappers
        self.cmo = self.I(talib.CMO, self.data.Close, timeperiod=14)
        
        # Bollinger Bands
        self.bb_upper = self.I(lambda close: talib.BBANDS(close, 20, 2, 2)[0], self.data.Close)
        self.bb_middle = self.I(lambda close: talib.BBANDS(close, 20, 2, 2)[1], self.data.Close)
        self.bb_lower = self.I(lambda close: talib.BBANDS(close, 20, 2, 2)[2], self.data.Close)
        
        # Bollinger Bandwidth calculations
        self.bb_bandwidth = self.I(lambda u, l: u - l, self.bb_upper, self.bb_lower)
        self.bb_squeeze = self.I(talib.MIN, self.bb_bandwidth, 20)
        
        # Volume filter
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20)
        
        # TICK calculations (assuming 'tick' column exists)
        self.tick_ma = self.I(talib.SMA, self.data.Tick, 5)
        
        print("🌙✨ MOON DEV INIT COMPLETE! Indicators ready for launch! 🚀")

    def next(self):
        current_bar = len(self.data) - 1
        
        # Entry conditions check
        cmo = self.cmo[-1]
        cmo_prev = self.cmo[-2] if len(self.cmo) > 1 else cmo
        volume_ok = self.data.Volume[-1] > self.volume_sma[-1]
        bb_squeeze = self.bb_bandwidth[-1] <= self.bb_squeeze[-1]
        tick_ok = (self.data.Tick[-1] > self.tick_ma[-1]) if self.position.is_short or not self.position else (self.data.Tick[-1] < self.tick_ma[-1])

        # Moon Dev themed debug prints 🌕
        if current_bar % 100 == 0:
            print(f"🌙 BAR {current_bar} | CMO: {cmo:.1f} | BB Width: {self.bb_bandwidth[-1]:.2f} | Volume Ratio: {self.data.Volume[-1]/self.volume_sma[-1]:.2f}x")

        # Close positions first
        if self.position:
            # Exit conditions for longs
            if self.position.is_long:
                if ((self.cmo[-2] > 30 and self.cmo[-1] <= 30) or  # Bearish crossunder replacement
                    self.data.Close[-1] < self.bb_lower[-1] or
                    self.position.duration >= 5):
                    
                    print(f"🌑 CLOSING LONG | CMO: {cmo:.1f} | Duration: {self.position.duration} bars ✨")
                    self.position.close()

            # Exit conditions for shorts
            elif self.position.is_short:
                if ((self.cmo[-2] < -30 and self.cmo[-1] >= -30) or  # Bullish crossover replacement
                    self.data.Close[-1] > self.bb_upper[-1] or
                    self.position.duration >= 5):
                    
                    print(f"🌑 CLOSING SHORT | CMO: {cmo:.1f} | Duration: {self.position.duration} bars ✨")
                    self.position.close()
        else:
            # Long entry with Moon Dev flare 🚀
            if (self.cmo[-2] <= 50 and self.cmo[-1] > 50) and bb_squeeze and volume_ok and self.data.Tick[-1] > self.tick_ma[-1]:
                sl = self.bb_lower[-1]
                risk = self.data.Close[-1] - sl
                size = int(round((self.equity * self.risk_per_trade) / risk))
                
                if size > 0 and risk > 0:
                    self.buy(size=size, sl=sl)