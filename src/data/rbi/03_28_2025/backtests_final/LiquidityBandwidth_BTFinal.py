Here's the complete fixed code with Moon Dev themed improvements and proper position sizing:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class LiquidityBandwidth(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌙 Moon Dev Indicator Setup
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='Swing High')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='Swing Low')
        
        # 🌌 Cosmic Bollinger Bands
        self.bb_upper = self.I(lambda c: talib.BBANDS(c, 20, 2, 2)[0], self.data.Close, name='BB Upper')
        self.bb_middle = self.I(lambda c: talib.BBANDS(c, 20, 2, 2)[1], self.data.Close, name='BB Middle')
        self.bb_lower = self.I(lambda c: talib.BBANDS(c, 20, 2, 2)[2], self.data.Close, name='BB Lower')
        
        # 📡 Bandwidth Scanner
        self.bb_width = self.I(lambda u, l, m: (u - l) / m, self.bb_upper, self.bb_lower, self.bb_middle, name='BB Width')
        self.bb_width_avg = self.I(talib.SMA, self.bb_width, 50, name='BB Width Avg')
        
        # ⚡ Momentum Thruster
        self.rsi = self.I(talib.RSI, self.data.Close, 14, name='RSI')
        
        print("🌙✨ Moon Dev Indicators Activated! Cosmic analysis systems online 🚀")
        print("🔭 Scanning for celestial trading opportunities...")

    def next(self):
        current_close = self.data.Close[-1]
        current_rsi = self.rsi[-1]
        equity = self.equity
        
        # 🌠 No overlapping trades
        if self.position:
            return
            
        # 🚀 Long Entry Conditions
        swing_high = self.swing_high[-1]
        price_distance = (swing_high - current_close) / swing_high
        if (0.005 <= price_distance <= 0.015 and 
            self.bb_width[-1] < 0.5 * self.bb_width_avg[-1] and 
            50 < current_rsi < 70):
            
            # 🛰️ Risk Management System
            stop_loss = swing_high * 1.003  # 0.3% beyond swing high
            risk_amount = equity * self.risk_per_trade
            risk_per_unit = abs(stop_loss - current_close)
            
            if risk_per_unit == 0:
                print("🌑 WARNING: Zero risk detected! Aborting launch sequence")
                return
                
            position_size = int(round(risk_amount / risk_per_unit))
            take_profit = current_close + (current_close - stop_loss)
            
            self.buy(size=position_size, sl=stop_loss, tp=take_profit)
            print(f"🚀🌕 LONG LAUNCH DETECTED! Size: {position_size:,} | Entry: {current_close:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
            print("🌌 Cosmic forces aligned for bullish trajectory!")

        # 🌑 Short Entry Conditions    
        swing_low = self.swing_low[-1]
        price_distance = (current_close - swing_low) / swing_low
        if (0.005 <= price_distance <= 0.015 and 
            self.bb_width[-1] < 0.5 * self.bb_width_avg[-1] and 
            30 < current_rsi < 50):
            
            # 🛰️ Risk Management System
            stop_loss = swing_low * 0.997  # 0.3% beyond swing low
            risk_amount = equity * self.risk_per_trade
            risk_per_unit = abs(current_close - stop_loss)
            
            if risk_per_unit == 0:
                print("🌑 WARNING: Zero risk detected! Aborting short dive")
                return
                
            position_size = int(round(risk_amount / risk_per_unit))
            take_profit = current_close - (stop_loss - current_close)
            
            self.sell(size=position_size, sl=stop_loss, tp=take_profit)
            print(f"🌑💫 SHORT DIVE INITIATED! Size: {position