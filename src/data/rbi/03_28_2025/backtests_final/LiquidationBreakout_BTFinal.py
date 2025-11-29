I'll debug the code while maintaining the original strategy logic. Here's the fixed version with Moon Dev themed improvements:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

class LiquidationBreakout(Strategy):
    # Strategy parameters
    risk_per_trade = 0.01  # 1% risk per trade
    atr_multiplier = 2
    liquidation_lookback = 20
    volume_multiplier = 1.5
    
    def init(self):
        # Calculate indicators using TA-Lib through self.I()
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.liquidation_lookback, name='SWING HIGH 🌊')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.liquidation_lookback, name='SWING LOW 🔻')
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VOL MA 📊')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR ⚡')
        
        print("🌙 Moon Dev Backtest Initialized! Ready for Launch! 🚀")
        print("✨ All indicators powered by pure TA-Lib magic! ✨")

    def next(self):
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        
        # Only trade when we have sufficient historical data
        if len(self.data) < self.liquidation_lookback + 5:
            return
            
        # Long entry condition
        if (current_close > self.swing_high[-2] and 
            current_volume > self.volume_ma[-1] * self.volume_multiplier):
            
            sl = self.swing_low[-2]
            risk_distance = current_close - sl
            atr_value = self.atr[-1]
            
            if risk_distance > 0 and atr_value > 0:
                position_size = self.calculate_position_size(risk_distance)
                tp = current_close + (self.atr_multiplier * atr_value)
                
                if not self.position:
                    print(f"🚀 LONG ENTRY! Breaking {self.swing_high[-2]:.2f} with volume surge!")
                    print(f"🌕 Moon Dev Position: Size={position_size}, SL={sl:.2f}, TP={tp:.2f}")
                    self.buy(size=position_size, sl=sl, tp=tp)
        
        # Short entry condition
        elif (current_close < self.swing_low[-2] and 
              current_volume > self.volume_ma[-1] * self.volume_multiplier):
            
            sl = self.swing_high[-2]
            risk_distance = sl - current_close
            atr_value = self.atr[-1]
            
            if risk_distance > 0 and atr_value > 0:
                position_size = self.calculate_position_size(risk_distance)
                tp = current_close - (self.atr_multiplier * atr_value)
                
                if not self.position:
                    print(f"🔻 SHORT ENTRY! Breaking {self.swing_low[-2]:.2f} with volume surge!")
                    print(f"🌑 Moon Dev Position: Size={position_size}, SL={sl:.2f}, TP={tp:.2f}")
                    self.sell(size=position_size, sl=sl, tp=tp)
        
        # Emergency exit check
        if self.position:
            if (self.position.is_long and current_close < self.swing_high[-2]) or \
               (self.position.is_short and current_close > self.swing_low[-2]):
                print(f"🚨 EMERGENCY EXIT! Price returned to liquidation zone!")
                print("🌘 Moon Dev Protection Protocol Activated!")
                self.position.close()

    def calculate_position_size(self, risk_distance):
        equity = self.equity
        risk_amount = equity * self.risk_per_trade
        position_size = risk_amount / risk_distance
        return int(round(position_size))  # Ensure whole number units

# Example usage (commented out for safety)
"""
if __name__ == "__main__":
    # Load your data here
    data = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
    
    # Run backtest
    bt = Backtest(data, LiquidationBreakout, commission=.002, margin=1.0)
    stats = bt.run()
    print("🌕 Moon Dev Backtest Complete! 🌕")
    print(stats)