```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
import numpy as np

class DivergentKeltner(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    atr_period = 20
    keltner_multiplier = 2.5
    netflow_ma_period = 30
    sar_acceleration = 0.02
    sar_maximum = 0.2
    
    def init(self):
        # Calculate indicators using TA-Lib with self.I()
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        
        # Keltner Channels
        self.upper_keltner = self.I(lambda: self.ema + self.keltner_multiplier * self.atr, name='Upper Keltner')
        self.lower_keltner = self.I(lambda: self.ema - self.keltner_multiplier * self.atr, name='Lower Keltner')
        
        # Parabolic SAR
        self.sar = self.I(talib.SAR, self.data.High, self.data.Low, 
                          acceleration=self.sar_acceleration, 
                          maximum=self.sar_maximum, name='SAR')
        
        # Netflow indicators
        self.netflow_ma = self.I(talib.SMA, self.data.df['netflow'], timeperiod=self.netflow_ma_period, name='Netflow MA')
        
        print("🌙 Moon Dev Indicators Activated! ✨")

    def next(self):
        current_close = self.data.Close[-1]
        current_sar = self.sar[-1]
        netflow_ma = self.netflow_ma[-1]
        
        # Moon Dev Risk Management Protocol 🚀
        atr_value = self.atr[-1] if len(self.atr) > 0 else 0
        
        if not self.position:
            # Long Entry Conditions
            if (current_close < self.lower_keltner[-1] and
                netflow_ma > 0 and  # Positive netflow divergence
                current_sar < current_close):
                
                sl_price = self.data.Low[-1] - atr_value
                risk_amount = self.equity * self.risk_per_trade
                position_size = int(round(risk_amount / (current_close - sl_price)))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=sl_price)
                    print(f"🌙 LONG Signal! Size: {position_size:,} @ {current_close:.2f} | SL: {sl_price:.2f} 🚀")

            # Short Entry Conditions        
            elif (current_close > self.upper_keltner[-1] and
                  netflow_ma < 0 and  # Negative netflow divergence
                  current_sar > current_close):
                
                sl_price = self.data.High[-1] + atr_value
                risk_amount = self.equity * self.risk_per_trade
                position_size = int(round(risk_amount / (sl_price - current_close)))
                
                if position_size > 0:
                    self.sell(size=position_size, sl=sl_price)
                    print(f"🌙 SHORT Signal! Size: {position_size:,} @ {current_close:.2f} | SL: {sl_price:.2f} 💥")

        else:
            # Exit Conditions
            if self.position.is_long:
                # SAR Reversal Exit
                if current_sar > current_close:
                    self.position.close()
                    print(f"🌙 Closing LONG (SAR Reversal) @ {current_close:.2f} 🛑")
                
                # Emergency Exits
                else:
                    upper_trigger = self.upper_keltner[-1] + 1.5 * atr_value
                    if self.data.High[-1] >= upper_trigger:
                        self.position.close()
                        print(f"🌙 Emergency LONG Exit (Price Breakout) @ {current_close:.2f} 🚨")
                    
                    if len(self.netflow_ma) >= 3 and all(x <= 0 for x in self.netflow_ma[-3:]):
                        self.position.close()
                        print(f"🌙 Emergency LONG Exit (Netflow Divergence Lost) @ {current_close:.2f} 💔")

            elif self.position.is_short:
                # SAR Reversal Exit
                if current_sar < current_close:
                    self.position.close()
                    print