```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import numpy as np
from backtesting import Strategy, Backtest

class DivergentVolume(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators using TA-Lib
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=50)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
        # MACD components
        def MACD_hist(close):
            macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            return hist
        self.macd_hist = self.I(MACD_hist, self.data.Close)
        
        # Volume indicators
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20)
        self.volume_ratio = self.I(lambda v, sma: v/sma, self.data.Volume, self.volume_sma)
        
        # Swing lows for divergence detection
        self.price_swing_low = self.I(talib.MIN, self.data.Low, 5)
        self.macd_swing_low = self.I(talib.MIN, self.macd_hist, 5)

    def next(self):
        # Moon Dev debug prints 🌙
        print(f"\n🌙 Moon Dev Debug - Bar {len(self.data)-1} ✨")
        print(f"Close: {self.data.Close[-1]:.2f}, EMA50: {self.ema50[-1]:.2f}")
        print(f"MACD Hist: {self.macd_hist[-1]:.4f}, Vol Ratio: {self.volume_ratio[-1]:.2f}")
        
        if not self.position:
            # Detect bullish divergence
            price_divergence = (self.price_swing_low[-1] < self.price_swing_low[-5] and 
                               self.macd_swing_low[-1] > self.macd_swing_low[-5])
            
            # Entry conditions
            vol_confirm = (self.volume_ratio[-1] > 1.2 and 
                          self.volume_ratio[-1] > self.volume_ratio[-2])
            trend_confirm = self.data.Close[-1] > self.ema50[-1]
            macd_trigger = self.macd_hist[-1] > 0 and self.macd_hist[-2] <= 0
            
            if price_divergence and vol_confirm and trend_confirm and macd_trigger:
                # Calculate position size
                risk_amount = self.equity * self.risk_percent
                atr_value = self.atr[-1]
                stop_loss = self.data.Close[-1] - 2 * atr_value
                risk_per_share = self.data.Close[-1] - stop_loss
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.buy(size=position_size)
                        print(f"🚀 LONG ENTRY: {position_size} shares @ {self.data.Close[-1]:.2f}")
                        print(f"🎯 TP: {self.data.Close[-1] + 3*atr_value:.2f}, 🛑 SL: {stop_loss:.2f}")
        else:
            # Update trailing stop
            current_high = max(self.data.High[-1], self.data.High[-2])
            new_trailing_stop = current_high - 2 * self.atr[-1]
            self.trailing_stop = max(new_trailing_stop, self.trailing_stop)
            
            # Exit conditions
            if self.data.Close[-1] >= (self.position.entry_price + 3*self.atr[-1]):
                self.sell()
                print("🎯 TAKE PROFIT REACHED!")
            elif self.data.Close[-1] <= self.trailing_stop:
                self.sell()
                print("🛑 TRAILING STOP TRIGGERED!")
            elif (self.macd_hist[-1] < 0 and self.macd_hist