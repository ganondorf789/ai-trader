```python
# 🌙✨ Moon Dev's VolumetricFibMACD Backtest Script 🚀📈

import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class VolumetricFibMACD(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌌 Calculate indicators using TA-Lib wrapped in self.I()
        # MACD components
        self.ema12 = self.I(talib.EMA, self.data.Close, 12)
        self.ema26 = self.I(talib.EMA, self.data.Close, 26)
        self.macd_line = self.ema12 - self.ema26
        self.signal_line = self.I(talib.EMA, self.macd_line, 9)
        self.signal_ema = self.I(talib.EMA, self.signal_line, 26)
        
        # Fibonacci swing points
        self.swing_high = self.I(talib.MAX, self.data.High, 20)
        self.swing_low = self.I(talib.MIN, self.data.Low, 20)
        
        # Volatility filter
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.atr_avg = self.I(talib.SMA, self.atr, 14)
        
    def next(self):
        # 🚨 Skip early bars without sufficient data
        if len(self.data.Close) < 20:
            return
            
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        prev_volume = self.data.Volume[-2]
        
        # 🌗 Calculate Fibonacci levels
        fib_range = self.swing_high[-1] - self.swing_low[-1]
        fib_382_long = self.swing_high[-1] - fib_range * 0.382
        fib_618_long = self.swing_high[-1] - fib_range * 0.618
        fib_382_short = self.swing_low[-1] + fib_range * 0.382
        
        # 🌪️ Volatility condition check
        atr_condition = self.atr[-1] > 1.5 * self.atr_avg[-1]
        
        # 💹 Long entry conditions
        if (not self.position and
            crossover(self.signal_line, self.signal_ema) and
            current_close <= fib_382_long and
            current_volume > prev_volume and
            atr_condition):
            
            risk_amount = self.equity * self.risk_percent
            stop_price = fib_618_long
            risk_distance = abs(current_close - stop_price)
            
            if risk_distance > 0:
                position_size = int(round(risk_amount / risk_distance))
                take_profit = current_close + 2 * risk_distance
                self.buy(size=position_size, sl=stop_price, tp=take_profit)
                print(f"🌙🚀 LONG ENTRY @ {current_close:.2f} | Size: {position_size} | Risk: {risk_distance:.2f} | Moon Power Activated! 🌕")
        
        # 📉 Short entry conditions
        elif (not self.position and
              crossover(self.signal_ema, self.signal_line) and
              current_close >= fib_382_short and
              current_volume > prev_volume and
              atr_condition):
            
            risk_amount = self.equity * self.risk_percent
            stop_price = self.swing_high[-1] - fib_range * 0.618  # Alternate SL method
            risk_distance = abs(current_close - stop_price)
            
            if risk_distance > 0:
                position_size = int(round(risk_amount / risk_distance))
                take_profit = current_close - 2 * risk_distance
                self.sell(size=position_size, sl=stop_price, tp=take_profit)
                print(f"🌙🌑 SHORT ENTRY @ {current_close:.2f} | Size: {position_size} | Risk: {risk_distance:.2f} | Dark Side Energy Engaged! 🌘")
        
        # 💎 Exit conditions for open positions
        if self.position.is_long and crossover(self.signal_ema, self.signal_line):
            self.position.close()
            print(f"🌙✨ LONG CLOSE @ {current_close:.2f} | Signal Reversal Detected! 🌠")
            
        elif self.position.is_short and crossover(self.signal_line, self.signal_ema):
            self.position.close()
            print(f"🌙✨ SH