I'll analyze and fix the code to ensure no backtesting.lib imports or functions are used. Here's the corrected version with Moon Dev themed improvements:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import numpy as np
from backtesting import Strategy, Backtest

class DynamicBreakoutEMA(Strategy):
    ema50_period = 50
    ema200_period = 200
    adx_period = 14
    atr_period = 14
    swing_high_period = 20
    risk_percent = 0.01
    
    def init(self):
        # Calculate core indicators using TA-Lib
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema50_period)
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema200_period)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_high_period)
        
    def next(self):
        # Moon Dev Debugging Variables 🌙
        entry_conditions = []
        
        if not self.position:
            # ======= ENTRY LOGIC =======
            # 1. EMA Golden Cross (replaced crossover function with direct comparison)
            ema_cross = (self.ema50[-2] < self.ema200[-2]) and (self.ema50[-1] > self.ema200[-1])
            entry_conditions.append(f"EMA Cross: {ema_cross}")
            
            # 2. ADX Strength Filter
            adx_strong = (self.adx[-1] > 25) and (self.adx[-1] > self.adx[-2])
            entry_conditions.append(f"ADX >25 & Rising: {adx_strong}")
            
            # 3. Price Breakout Filter
            price_breakout = self.data.Close[-1] > self.swing_high[-1]
            entry_conditions.append(f"Price > Swing High: {price_breakout}")
            
            if all([ema_cross, adx_strong, price_breakout]):
                # ======= RISK MANAGEMENT =======
                atr_value = self.atr[-1]
                entry_price = self.data.Close[-1]
                stop_loss_price = entry_price - 2 * atr_value
                risk_per_share = entry_price - stop_loss_price
                position_size = int(round((self.equity * self.risk_percent) / risk_per_share))
                
                if position_size > 0:
                    # Store initial stop and trail status
                    self.initial_stop = stop_loss_price
                    self.trailing_active = False
                    
                    # Execute entry with initial stop loss
                    self.buy(size=position_size)
                    print(f"\n🌙✨🚀 MOON ENTRY SIGNAL @ {entry_price:.2f}")
                    print(f"Size: {position_size} | Risk: {self.risk_percent*100}%")
                    print(f"SL: {stop_loss_price:.2f} ({2*atr_value:.2f} risk)")
                    print("Entry Conditions:", " | ".join(entry_conditions))
        else:
            # ======= EXIT LOGIC =======
            current_low = self.data.Low[-1]
            current_high = self.data.High[-1]
            entry_price = self.position.entry_price
            
            # 1. Initial Stop Loss Check
            if not self.trailing_active and current_low <= self.initial_stop:
                self.position.close()
                print(f"\n🌧️🌧️🌧️ INITIAL STOP HIT @ {self.data.Close[-1]:.2f}")
                print(f"Equity: {self.equity:.2f} | PnL: {self.position.pl:.2f}")
            
            # 2. Trailing Stop Activation & Management
            elif not self.trailing_active and current_high >= entry_price * 1.05:
                self.trailing_active = True
                self.peak_price = current_high
                self.trailing_stop = self.peak_price * 0.95
                print(f"\n🚀🚀🚀 TRAILING STOP ACTIVATED @ {self.trailing_stop:.2f}")
            
            # 3. Trailing Stop Update
            if self.t