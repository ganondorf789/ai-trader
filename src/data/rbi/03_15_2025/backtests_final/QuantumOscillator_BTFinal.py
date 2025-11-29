Here's the fixed and complete backtest code with Moon Dev themed debug prints and proper position sizing implementation:

```python
# 🌙 Moon Dev's QuantumOscillator Backtest 🌙
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

class QuantumOscillator(Strategy):
    # Strategy parameters
    risk_per_trade = 0.01  # 1% risk per trade
    atr_period = 14
    qvi_period = 20
    rsi_period = 14
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators using TA-Lib with self.I()
        # Quantum Volatility Index (QVI)
        self.max_high = self.I(talib.MAX, self.data.High, timeperiod=self.qvi_period, name='MAX_HIGH')
        self.min_low = self.I(talib.MIN, self.data.Low, timeperiod=self.qvi_period, name='MIN_LOW')
        self.qvi = self.I(lambda: ((self.max_high - self.min_low) / self.min_low * 100), name='QVI')
        
        # Adaptive RSI
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period, name='RSI')
        
        # Average True Range
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period, name='ATR')
        
        print("🌌 QuantumOscillator Strategy Initialized! 🚀")

    def next(self):
        current_close = self.data.Close[-1]
        current_qvi = self.qvi[-1]
        current_rsi = self.rsi[-1]
        current_atr = self.atr[-1]
        
        # Moon-themed debug prints ✨
        print(f"🌙 Current Close: {current_close:.2f} | QVI: {current_qvi:.2f} | RSI: {current_rsi:.2f} | ATR: {current_atr:.2f}")
        
        # Risk Management Calculations
        equity = self.equity
        risk_amount = equity * self.risk_per_trade
        position_size = risk_amount / (0.75 * current_atr)
        
        # Volatility adjustment
        if current_qvi > 70:
            position_size *= 0.5
            print("⚡ Volatility Alert! Reducing position size by 50%")
        
        # Ensure position size is a whole number
        position_size = int(round(position_size))
        if position_size < 1:
            print("🌑 Position size too small, skipping trade")
            return  # Skip trade if size too small
            
        # Entry Logic
        if not self.position:
            # Bullish Divergence Check
            bull_div = (self.data.Low[-1] > self.data.Low[-2] and 
                       self.rsi[-1] < self.rsi[-2])
            
            # Bearish Divergence Check
            bear_div = (self.data.High[-1] < self.data.High[-2] and 
                        self.rsi[-1] > self.rsi[-2])
            
            # Long Entry Conditions
            if (current_qvi < 30 and
                self.rsi[-2] < 35 and self.rsi[-1] > 35 and  # Replaced crossover
                bull_div):
                
                sl = current_close - 0.75 * current_atr
                tp = current_close + 1.5 * current_atr
                self.buy(size=position_size, sl=sl, tp=tp)
                print(f"🚀 LONG ENTRY! Size: {position_size} | SL: {sl:.2f} | TP: {tp:.2f}")
            
            # Short Entry Conditions
            elif (current_qvi > 70 and
                  self.rsi[-2] > 65 and self.rsi[-1] < 65 and  # Replaced crossunder
                  bear_div):
                
                sl = current_close + 0.75 * current_atr
                tp = current_close - 1.5 * current_atr
                self.sell(size=position_size, sl=sl, tp=tp)
                print(f"🌧️ SHORT ENTRY! Size: {position_size} | SL: {sl:.2f} | TP: {tp:.2f}")