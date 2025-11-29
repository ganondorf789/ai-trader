```python
# -*- coding: utf-8 -*-
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib
import pandas_ta as ta

class VortexChaikinTrail(Strategy):
    def init(self):
        # Calculate indicators with proper TA-Lib/pandas_ta usage
        self.current_bar = 0  # For tracking time-based exits
        
        # Vortex Indicator (VI+ and VI-)
        vortex = self.data.ta.vortex(length=14)
        self.vi_plus = self.I(lambda: vortex['VORTICSp_14'], name='VI+')
        self.vi_minus = self.I(lambda: vortex['VORTICSm_14'], name='VI-')
        
        # Chaikin Money Flow (CMF)
        self.cmf = self.I(talib.CMF, self.data.High, self.data.Low, 
                         self.data.Close, self.data.Volume, 20, name='CMF')
        
        # Parabolic SAR (PSAR)
        self.psar = self.I(talib.SAR, self.data.High, self.data.Low, 
                          acceleration=0.02, maximum=0.2, name='PSAR')
        
        # Swing High/Low for backup stops
        self.swing_high = self.I(talib.MAX, self.data.High, 20, name='Swing High')
        self.swing_low = self.I(talib.MIN, self.data.Low, 20, name='Swing Low')

    def next(self):
        self.current_bar += 1
        
        # Moon Dev debug prints 🌙✨
        if len(self.data) % 100 == 0:
            print(f"🌙 Processing bar {self.current_bar} | Equity: {self.equity:,.2f} ✨")
        
        # Check exit conditions first
        if self.position:
            # PSAR trailing stop
            if self.position.is_long and self.data.Close[-1] < self.psar[-1]:
                self.position.close()
                print(f"🌙✨ Closing LONG at {self.data.Close[-1]:.2f} (PSAR Trail)")
            elif self.position.is_short and self.data.Close[-1] > self.psar[-1]:
                self.position.close()
                print(f"🌙✨ Closing SHORT at {self.data.Close[-1]:.2f} (PSAR Trail)")
            
            # Time-based exit after 10 bars
            for trade in self.trades:
                if not trade.is_closed and (self.current_bar - trade.entry_bar) >= 10:
                    trade.close()
                    print(f"🌙⏰ Time Exit at {self.data.Close[-1]:.2f}")

        # Entry logic when flat
        if not self.position:
            cmf_val = self.cmf[-1]
            
            # Long entry: VI+ crosses above VI- and CMF > 0.05
            if crossover(self.vi_plus, self.vi_minus) and cmf_val > 0.05:
                entry_price = self.data.Close[-1]
                sl_price = self.swing_low[-1]
                risk = entry_price - sl_price
                
                if risk > 0:
                    risk_pct = 0.01  # 1% risk
                    risk_amount = self.equity * risk_pct
                    size = int(round(risk_amount / risk))
                    if size > 0:
                        self.buy(size=size, stop=sl_price)
                        print(f"🚀🌙 LONG Entry: {entry_price:.2f} | Size: {size} | SL: {sl_price:.2f}")
            
            # Short entry: VI- crosses above VI+ and CMF < -0.05
            if crossover(self.vi_minus, self.vi_plus) and cmf_val < -0.05:
                entry_price = self.data.Close[-1]
                sl_price = self.swing_high[-1]
                risk = sl_price - entry_price
                
                if risk > 0:
                    risk_pct = 0.01  # 1% risk
                    risk_amount = self.equity * risk_pct
                    size = int(round(risk_amount / risk))
                    if size > 0:
                        self.sell(size=size, stop=sl_price)
                        print(f"🌙🚀 SHORT Entry: {entry_price:.2f} | Size: {size} | SL: {sl_price:.2f}")

# Data handling with Moon Dev standards 🌙📊
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-