Here's the fixed and complete backtest code with all Moon Dev debug enhancements 🌙✨:

```python
# moon_dev_backtest.py 🌙✨
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

class DynamicBandSqueeze(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌕
    adx_period = 14
    bb_period = 20
    keltner_multiplier = 1.5
    
    def init(self):
        # Moon Dev Indicator Suite 🚀
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)
        
        # Bollinger Bands Calculation ✨
        self.bb_upper = self.I(lambda: talib.BBANDS(self.data.Close, timeperiod=self.bb_period, nbdevup=2, nbdevdn=2)[0])
        self.bb_middle = self.I(lambda: talib.BBANDS(self.data.Close, timeperiod=self.bb_period, nbdevup=2, nbdevdn=2)[1])
        self.bb_lower = self.I(lambda: talib.BBANDS(self.data.Close, timeperiod=self.bb_period, nbdevup=2, nbdevdn=2)[2])
        
        # Keltner Channel Magic 🌙
        self.keltner_mid = self.I(talib.EMA, self.data.Close, timeperiod=self.bb_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.bb_period)
        self.keltner_upper = self.I(lambda: self.keltner_mid + self.keltner_multiplier * self.atr)
        self.keltner_lower = self.I(lambda: self.keltner_mid - self.keltner_multiplier * self.atr)
        
        # Swing High/Low Detection 🌄
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.bb_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.bb_period)
        
        print("🌙 Moon Dev Indicators Activated! ✨ Lunar Algorithms Online 🌗")

    def next(self):
        # Moon Dev Position Management 🌗
        for trade in self.trades:
            if trade.is_long and self.data.Close[-1] < self.keltner_mid[-1]:
                trade.close()
                print(f"🌙 Exit Long: Price {self.data.Close[-1]:.2f} < Keltner Mid {self.keltner_mid[-1]:.2f}")
            elif trade.is_short and self.data.Close[-1] > self.keltner_mid[-1]:
                trade.close()
                print(f"🌑 Exit Short: Price {self.data.Close[-1]:.2f} > Keltner Mid {self.keltner_mid[-1]:.2f}")

        # Moon Dev Entry Logic 🌟
        if len(self.data) < 20 or self.adx[-1] < 25:
            return
            
        # Calculate Bollinger Bandwidth 🌈
        bb_width = (self.bb_upper[-3:] - self.bb_lower[-3:]) / self.bb_middle[-3:]
        contraction = all(bb_width <= 0.02)
        
        if not self.position and contraction:
            # Long Entry Signal 🚀
            if self.data.Close[-1] > self.bb_upper[-1]:
                sl = min(self.swing_low[-1], self.bb_lower[-1])
                risk = self.data.Close[-1] - sl
                if risk > 0:
                    size = int(round((self.equity * self.risk_pct) / risk))
                    self.buy(size=size, sl=sl, tp=self.data.Close[-1] + 2*risk)
                    print(f"🚀 LONG Entry: {self.data.Close[-1]:.2f} | Size: {size} | R/R: 1:2 🌕")
            
            # Short Entry Signal 🌑
            elif self.data.Close[-1] < self.bb_lower[-1]:
                sl = max(self.swing_high[-1], self.bb_upper[-1])
                risk = sl - self.data.Close[-1]
                if risk > 0:
                    size = int(round((self.equity * self.risk_pct) / risk))
                    self.sell(size=size, sl=sl, tp=self.data.Close[-1]