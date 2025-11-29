Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed debug prints added:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class SqueezeSpikeTrail(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    trailing_atr_multiplier = 2
    
    def init(self):
        # 🌙 MOON DEV INDICATOR INITIALIZATION ✨
        def bbands(close):
            upper, _, lower = talib.BBANDS(close, 
                timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            return upper, lower
            
        self.bb_upper, self.bb_lower = self.I(bbands, self.data.Close)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 14)
        
        self.trailing_high = None
        self.trailing_low = None

    def next(self):
        # 🌙 MOON DEV CORE LOGIC 🚀
        if len(self.data.Close) < 34:  # 20+14 period warmup
            return
            
        current_close = self.data.Close[-1]
        current_upper = self.bb_upper[-1]
        current_lower = self.bb_lower[-1]
        current_volume = self.data.Volume[-1]
        
        # ✨ SQUEEZE CALCULATION 🌪️
        squeeze = (current_upper - current_lower)/current_close <= 0.015
        
        # 📈 VOLUME SPIKE DETECTION 🚨
        volume_spike = current_volume >= 1.5 * self.volume_sma[-1]
        
        # 🚦 ENTRY CONDITIONS CHECK ✅
        long_trigger = current_close > current_upper
        short_trigger = current_close < current_lower
        
        # 💼 RISK MANAGEMENT CALCULATIONS 💸
        risk_amount = self.equity * self.risk_percent
        atr_value = self.atr[-1]
        
        if not self.position:
            # 🌙 LONG ENTRY LOGIC 🚀
            if all([squeeze, volume_spike, long_trigger]):
                stop_distance = self.trailing_atr_multiplier * atr_value
                if stop_distance <= 0:
                    return
                
                position_size = int(round(risk_amount / stop_distance))
                if position_size > 0:
                    self.buy(size=position_size, 
                           sl=current_close - stop_distance)
                    self.trailing_high = self.data.High[-1]
                    print(f"🚀 MOON DEV LONG ENTRY! ✨ Size: {position_size} | Entry: {current_close:.2f} | SL: {current_close - stop_distance:.2f}")

            # 🌑 SHORT ENTRY LOGIC 🌪️
            elif all([squeeze, volume_spike, short_trigger]):
                stop_distance = self.trailing_atr_multiplier * atr_value
                if stop_distance <= 0:
                    return
                
                position_size = int(round(risk_amount / stop_distance))
                if position_size > 0:
                    self.sell(size=position_size,
                            sl=current_close + stop_distance)
                    self.trailing_low = self.data.Low[-1]
                    print(f"🌑 MOON DEV SHORT ENTRY! 🌪️ Size: {position_size} | Entry: {current_close:.2f} | SL: {current_close + stop_distance:.2f}")
        else:
            # 🛑 TRAILING STOP MANAGEMENT 🎢
            if self.position.is_long:
                if self.data.High[-1] > self.trailing_high:
                    self.trailing_high = self.data.High[-1]
                    new_sl = self.trailing_high - (self.trailing_atr_multiplier * atr_value)
                    if new_sl > self.position.sl:
                        self.position.sl = new_sl
                        print(f"✨ LONG TRAILING SL UPDATED 🌙 New SL: {new_sl:.2f}")

            elif self.position.is_short:
                if self.data.Low[-1] < self.trailing_low:
                    self.trailing_low = self.data.Low[-1]
                    new_sl = self.trailing_low + (self.trailing_atr_multiplier * atr