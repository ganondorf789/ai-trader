I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete fixed version with Moon Dev themed debug prints:

```python
# 🌙 Moon Dev's VoltaicSqueeze Backtest 🌙
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

class VoltaicSqueeze(Strategy):
    def init(self):
        # 🌙 Indicator Calculation Section ✨
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        # Calculate Bollinger Bands
        upper_bb, middle_bb, lower_bb = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        self.upper_bb = self.I(lambda: upper_bb, name='UpperBB')
        self.middle_bb = self.I(lambda: middle_bb, name='MiddleBB')
        self.lower_bb = self.I(lambda: lower_bb, name='LowerBB')
        
        # Calculate Bollinger Bandwidth Squeeze
        bb_width = (upper_bb - lower_bb) / middle_bb
        self.bb_width = self.I(lambda: bb_width, name='BBWidth')
        self.bb_width_min = self.I(lambda: talib.MIN(bb_width, 10), name='BBWidthMin')
        
        # Calculate Keltner Channels
        ema20 = talib.EMA(close, 20)
        atr20 = talib.ATR(high, low, close, 20)
        upper_kc = ema20 + 1.5 * atr20
        lower_kc = ema20 - 1.5 * atr20
        self.upper_kc = self.I(lambda: upper_kc, name='UpperKC')
        self.lower_kc = self.I(lambda: lower_kc, name='LowerKC')
        
        # Calculate ATR for risk management
        self.atr14 = self.I(lambda: talib.ATR(high, low, close, 14), name='ATR14')
        
        print("🌙 MOON DEV INDICATORS LOADED! ✨")

    def next(self):
        # 🌙 Moon Dev Debug Prints ✨
        if len(self.data) % 100 == 0:
            print(f"🌙 PROCESSING BAR {len(self.data)}/{len(self.data.df)} - MOON DEV POWER ACTIVE! 🚀")
        
        # 🌙 Entry Logic Section ✨
        if not self.position:
            # Long Entry Condition
            squeeze_long = (self.bb_width[-1] <= self.bb_width_min[-1]) and \
                          (self.data.Close[-1] > self.upper_kc[-1])
            
            # Short Entry Condition
            squeeze_short = (self.bb_width[-1] <= self.bb_width_min[-1]) and \
                           (self.data.Close[-1] < self.lower_kc[-1])
            
            if squeeze_long:
                # 🌙 Risk Management Calculation ✨
                atr_value = self.atr14[-1]
                equity = self.broker.equity
                risk_amount = 0.02 * equity
                position_size = int(round(risk_amount / atr_value))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    self.entry_price = self.data.Close[-1]
                    self.highest_high = self.data.High[-1]
                    print(f"🌙 LONG SIGNAL TRIGGERED! 🚀 | Size: {position_size} | Entry: {self.entry_price:.2f} | ATR14: {atr_value:.2f}")
            
            elif squeeze_short:
                atr_value = self.atr14[-1]
                equity = self.broker.equity
                risk_amount = 0.02 * equity
                position_size = int(round(risk_amount / atr_value))
                
                if position_size > 0:
                    self.sell(size=position_size)
                    self.entry_price = self.data.Close[-1]
                    self.lowest_low = self.data.Low[-1]
                    print(f"🌙 SHORT SIGNAL TRIGGERED! 🌪️ | Size: {position_size} | Entry: {self.entry_price:.2f} | ATR14: {atr_value:.2f}")
        
        # 🌙 Exit Logic Section ✨
        else:
            if self.position.is_long:
                self.highest_high = max(self.highest_high, self.data.High[-1])
                trail_stop = self.highest_high - 2 * self.atr14[-