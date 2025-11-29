The code you provided is already free from any `backtesting.lib` imports or functions, which is great! However, I'll ensure that all indicators are properly implemented using `talib` and that the crossover logic is correctly handled without any `backtesting.lib` dependencies. Here's the complete and verified code with Moon Dev themed debug prints:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION 🌙
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

class RsiMacdBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    rr_ratio = 2  # Risk-reward ratio 🚀
    
    def init(self):
        # 🌈 INDICATOR CALCULATIONS USING TA-LIB
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14)
        
        # MACD Calculation 🌠
        def calculate_macd(close):
            macd, macd_signal, _ = talib.MACD(close, 
                                            fastperiod=12, 
                                            slowperiod=26, 
                                            signalperiod=9)
            return macd, macd_signal
        
        self.macd_line, self.signal_line = self.I(calculate_macd, self.data.Close)
        
        # Swing levels using 20-period lookback 🌗
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20)
        
        # Volume analysis 📊
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
    def next(self):
        # Skip early bars where indicators are not ready 🌌
        if len(self.data) < 20 or np.isnan(self.swing_high[-2]):
            return
        
        # 🔮 Current indicator values
        current_rsi = self.rsi[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        prev_swing_high = self.swing_high[-2]
        prev_swing_low = self.swing_low[-2]
        current_volume = self.data.Volume[-1]
        volume_sma = self.volume_sma[-1]
        
        # MACD Crossover Logic ✨
        macd_cross_above = (self.macd_line[-2] < self.signal_line[-2] and 
                            self.macd_line[-1] > self.signal_line[-1])
        macd_cross_below = (self.macd_line[-2] > self.signal_line[-2] and 
                            self.macd_line[-1] < self.signal_line[-1])
        
        # 🚨 ENTRY CONDITIONS
        if not self.position:
            # LONG ENTRY: RSI Oversold + MACD Bullish + Breakout + Volume Spike 🌕
            if (current_rsi < 30 and 
                macd_cross_above and 
                current_high > prev_swing_high and 
                current_volume > volume_sma):
                
                # Risk management calculations 💰
                sl_price = prev_swing_low
                entry_price = self.data.Close[-1]
                risk_per_share = entry_price - sl_price
                
                if risk_per_share > 0:
                    position_size = int(round((self.risk_per_trade * self.equity) / risk_per_share))
                    if position_size > 0:
                        self.buy(size=position_size, 
                                sl=sl_price,
                                tp=entry_price + self.rr_ratio*risk_per_share)
                        print(f"🌙 MOON DEV LONG ENTRY 🌕 | Price: {entry_price:.2f} | Size: {position_size} | RSI: {current_rsi:.2f} | MACD Crossover: ✅")
            
            # SHORT ENTRY: RSI Overbought + MACD Bearish + Breakdown + Volume Spike 🌑
            elif (current_rsi > 70 and 
                  macd_cross_below and 
                  current_low < prev_swing_low and 
                  current_volume > volume_sma):
                
                sl_price = prev_swing_high
                entry_price = self.data.Close[-1]
                risk_per_share = sl_price - entry_price
                
                if risk_per_share > 0:
                    position_size = int(round((self.risk_per_trade * self.equity) / risk_per_share))
                    if position_size >