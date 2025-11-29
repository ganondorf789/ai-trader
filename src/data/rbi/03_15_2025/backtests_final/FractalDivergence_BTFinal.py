Here's the fixed and complete code with Moon Dev themed debug prints and proper position sizing implementation:

```python
from backtesting import Strategy, Backtest
import pandas as pd
import talib as ta
import numpy as np

class FractalDivergence(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    atr_period = 14
    rsi_period = 14
    ema_period = 50
    
    def init(self):
        # Clean data columns already handled in preprocessing
        self.rsi = self.I(ta.RSI, self.data.Close, self.rsi_period)
        self.ema50 = self.I(ta.EMA, self.data.Close, self.ema_period)
        self.atr = self.I(ta.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        # Fractal detection (5-period window)
        self.max_high = self.I(ta.MAX, self.data.High, 5)
        self.min_low = self.I(ta.MIN, self.data.Low, 5)
        
        # Track fractal states
        self.prev_bullish_low = None
        self.prev_bullish_rsi = None
        self.prev_bearish_high = None
        self.prev_bearish_rsi = None
        
    def next(self):
        if len(self.data) < 5:  # Ensure enough data for fractal confirmation
            return
        
        # Check for fractals two bars ago 🌙
        current_max_high = self.max_high[-1]
        current_min_low = self.min_low[-1]
        
        # Bearish fractal check (two bars ago)
        bearish_fractal = self.data.High[-3] == current_max_high
        # Bullish fractal check (two bars ago)
        bullish_fractal = self.data.Low[-3] == current_min_low
        
        # Update fractal tracking
        if bearish_fractal:
            self.prev_bearish_high = self.data.High[-3]
            self.prev_bearish_rsi = self.rsi[-3]
        if bullish_fractal:
            self.prev_bullish_low = self.data.Low[-3]
            self.prev_bullish_rsi = self.rsi[-3]
        
        # Check divergences ✨
        bullish_div = False
        bearish_div = False
        
        if bullish_fractal and self.prev_bullish_low is not None:
            bullish_div = (self.data.Low[-3] < self.prev_bullish_low and 
                         self.rsi[-3] > self.prev_bullish_rsi)
        
        if bearish_fractal and self.prev_bearish_high is not None:
            bearish_div = (self.data.High[-3] > self.prev_bearish_high and 
                          self.rsi[-3] < self.prev_bearish_rsi)
        
        # EMA slope calculation 🚀
        ema_slope_up = self.ema50[-1] < self.ema50[0]
        ema_slope_down = self.ema50[-1] > self.ema50[0]
        
        # RSI filter
        rsi_valid = 30 < self.rsi[-1] < 70
        
        # Entry logic 🌙✨
        if not self.position:
            # Long entry
            if bullish_div and self.data.Close[-1] > self.data.High[-3] and ema_slope_up and rsi_valid:
                sl = self.data.Low[-3] - 1.5 * self.atr[-1]
                risk = self.risk_percent * self.equity
                risk_per_unit = self.data.Close[-1] - sl
                if risk_per_unit <= 0: 
                    print("🌙 WARNING: Invalid risk per unit for long trade")
                    return
                size = int(round(risk / risk_per_unit))
                if size > 0:
                    tp = self.data.Close[-1] + 2 * risk_per_unit
                    self.buy(size=size, sl=sl, tp=tp)
                    print(f"🌙✨🚀 BULLISH BREAKOUT! Long {size} units at {self.data.Close[-1]:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
            
            # Short entry
            elif bearish_div and self.data.Close[-1] < self.data.Low[-3] and ema_slope_down and rsi_valid:
                sl = self.data.High[-3] + 1.5 * self.atr[-1]
                risk = self.risk