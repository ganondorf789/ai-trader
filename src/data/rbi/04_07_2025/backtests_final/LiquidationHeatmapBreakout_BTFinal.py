Here's the debugged version of your LiquidationHeatmapBreakout strategy with Moon Dev themed fixes:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

class LiquidationHeatmapBreakout(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌙
    max_exposure = 0.05  # 5% max portfolio exposure ✨
    atr_period = 14
    consolidation_period = 20
    
    def init(self):
        # Moon-powered indicators initialization 🌕
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.atr_avg = self.I(talib.SMA, self.atr, timeperiod=20)
        
        # Lunar consolidation boundaries 🌘
        self.upper_bound = self.I(talib.MAX, self.data.High, timeperiod=self.consolidation_period)
        self.lower_bound = self.I(talib.MIN, self.data.Low, timeperiod=self.consolidation_period)
        
        # Cosmic volume analysis 🌌
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
        self.trade_entry_bar = 0  # Track entry bar for time-based exits

    def next(self):
        # Wait for enough data to form indicators 🌙
        if len(self.data) < self.consolidation_period:
            return
            
        current_atr = self.atr[-1]
        upper = self.upper_bound[-1]
        lower = self.lower_bound[-1]
        close = self.data.Close[-1]
        volume = self.data.Volume[-1]

        # Moon Dev volatility contraction check ✨
        volatility_contraction = current_atr < 0.7 * self.atr_avg[-1]
        
        # Stellar breakout triggers 🌠
        long_trigger = upper + 2 * current_atr
        short_trigger = lower - 2 * current_atr

        if not self.position:
            # 🌕 Long entry constellation
            if close > long_trigger and volatility_contraction and volume > self.volume_sma[-1]:
                sl = lower - 0.5 * current_atr
                risk = close - sl
                position_size = self.calculate_position_size(risk)
                
                if position_size > 0:
                    self.buy(size=position_size, sl=sl, tp=close + 3*current_atr)
                    self.trade_entry_bar = len(self.data)
                    print(f"🌕 MOON SHOT! Long entry at {close:.2f} ✨")
                    print(f"   🛡️ SL: {sl:.2f} | 🎯 TP: {close + 3*current_atr:.2f} | 🚀 Size: {position_size}")

            # 🌑 Short entry constellation
            elif close < short_trigger and volatility_contraction and volume > self.volume_sma[-1]:
                sl = upper + 0.5 * current_atr
                risk = sl - close
                position_size = self.calculate_position_size(risk)
                
                if position_size > 0:
                    self.sell(size=position_size, sl=sl, tp=close - 3*current_atr)
                    self.trade_entry_bar = len(self.data)
                    print(f"🌑 DARK SIDE! Short entry at {close:.2f} 🌘")
                    print(f"   🛡️ SL: {sl:.2f} | 🎯 TP: {close - 3*current_atr:.2f} | 🚀 Size: {position_size}")
        else:
            # Time-based exit (5 bars) ⏳🌙
            if (len(self.data) - self.trade_entry_bar) >= 5:
                self.position.close()
                print(f"🕒 TIME'S UP! Closing position at {close:.2f} 🌙")

    def calculate_position_size(self, risk_per_trade):
        """Moon-sized risk management calculation 🌙✨"""
        if risk_per_trade <= 0:
            return 0
            
        risk_amount = self.equity * self.risk_percent
        size_by_risk = risk_amount / risk_per_trade
        
        max_size_by_exposure = (self.equity * self.max_exposure) / self.data.Close[-1]
        
        # Ensure whole number position sizes for crypto trading 🚀
        position_size = min(size_by_risk, max_size_by_exposure)
        return round(position_size)  # Round to nearest whole number