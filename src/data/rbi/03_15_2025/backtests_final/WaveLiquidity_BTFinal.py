I'll help you fix the code while maintaining the original strategy logic. Here's the corrected version with Moon Dev themed debug prints and proper formatting:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class WaveLiquidity(Strategy):
    timeperiod = 20  # Swing period
    risk_percent = 0.01  # 1% risk per trade
    volume_spike_multiplier = 1.5  # Volume spike threshold
    
    def init(self):
        # Core Indicators 🌊
        self.swing_high = self.I(talib.MAX, self.data.High, self.timeperiod)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.timeperiod)
        self.engulfing = self.I(talib.CDLENGULFING, 
                              self.data.Open, 
                              self.data.High, 
                              self.data.Low, 
                              self.data.Close)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, self.timeperiod)
        
        print("🌙✨ WaveLiquidity Strategy Activated! Scanning for moon-bound opportunities... 🚀")

    def next(self):
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        
        # Only trade once per bar and when indicators are valid
        if len(self.data) < self.timeperiod or self.position:
            return

        # Calculate Fibonacci levels 📐
        swing_high = self.swing_high[-1]
        swing_low = self.swing_low[-1]
        fib_diff = swing_high - swing_low
        levels = {
            '382': swing_high - fib_diff * 0.382,
            '500': swing_high - fib_diff * 0.5,
            '618': swing_high - fib_diff * 0.618
        }

        # Check Fib proximity (1% tolerance) 🎯
        fib_zone = any(abs(current_close - lvl) < current_close * 0.01 
                      for lvl in levels.values())
        
        # Volume validation 🔊
        volume_spike = current_volume > self.volume_sma[-1] * self.volume_spike_multiplier
        
        # Engulfing pattern detection 🕯️
        bullish_engulf = self.engulfing[-1] == 100
        bearish_engulf = self.engulfing[-1] == -100

        # Risk management calculations 🔒
        equity = self.equity
        risk_amount = equity * self.risk_percent

        # Long Entry Logic 🟢
        if fib_zone and bullish_engulf and volume_spike:
            sl = swing_low
            risk_per_unit = current_close - sl
            
            if risk_per_unit > 0:
                position_size = int(round(risk_amount / risk_per_unit))
                tp = current_close + 2 * risk_per_unit  # 1:2 RR
                
                if position_size > 0:
                    self.buy(size=position_size, sl=sl, tp=tp)
                    print(f"🌙🚀 BULLISH REVERSAL DETECTED! Entry: {current_close:.2f}")
                    print(f"Size: {position_size} | SL: {sl:.2f} | TP: {tp:.2f} ✨")

        # Short Entry Logic 🔴
        elif fib_zone and bearish_engulf and volume_spike:
            sl = swing_high
            risk_per_unit = sl - current_close
            
            if risk_per_unit > 0:
                position_size = int(round(risk_amount / risk_per_unit))
                tp = current_close - 2 * risk_per_unit  # 1:2 RR
                
                if position_size > 0:
                    self.sell(size=position_size, sl=sl, tp=tp)
                    print(f"🌙🌧️ BEARISH REVERSAL DETECTED! Entry: {current_close:.2f}")
                    print(f"Size: {position_size} | SL: {sl:.2f} | TP: {tp:.2f} ✨")

# Data preparation 📂
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    '