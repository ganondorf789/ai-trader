Here's the fully debugged code with Moon Dev themed improvements while preserving the original strategy logic:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

class VortexMomentum(Strategy):
    # Strategy parameters
    cmo_period = 14
    vi_period = 14
    volume_ma_period = 20
    atr_period = 14
    risk_pct = 0.01  # 1% risk per trade
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators
        # Vortex Indicator using pandas_ta
        vi = self.I(ta.vortex, 
                   high=self.data.High, 
                   low=self.data.Low, 
                   close=self.data.Close, 
                   length=self.vi_period,
                   name=['VI_plus', 'VI_minus'])
        self.vi_plus = vi[0]  # First line is VI+
        self.vi_minus = vi[1]  # Second line is VI-
        
        # Chande Momentum Oscillator
        self.cmo = self.I(talib.CMO, self.data.Close, timeperiod=self.cmo_period)
        
        # Volume MA
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_ma_period)
        
        # Swing High/Low using MAX/MIN
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20)
        
        # ATR for trailing stop
        self.atr = self.I(talib.ATR, 
                         self.data.High, 
                         self.data.Low, 
                         self.data.Close, 
                         timeperiod=self.atr_period)
        
        print("🌙✨ VortexMomentum Strategy Initialized with Moon Dev Power!")
        
    def next(self):
        current_close = self.data.Close[-1]
        
        # Moon Dev Debug Prints
        print(f"\n🌙 [Step {len(self.data)}] Price: {current_close:.2f}")
        print(f"   VI+:{self.vi_plus[-1]:.4f} | VI-:{self.vi_minus[-1]:.4f}")
        print(f"   CMO:{self.cmo[-1]:.2f} | Vol:{self.data.Volume[-1]:.2f} vs MA:{self.volume_ma[-1]:.2f}")
        
        if not self.position:
            # Long Entry Conditions
            if ((self.vi_plus[-2] < self.vi_minus[-2] and self.vi_plus[-1] > self.vi_minus[-1]) and \
               (self.cmo[-1] < -50) and \
               (self.data.Volume[-1] > self.volume_ma[-1])):
                
                stop_price = self.swing_low[-1]
                risk_per_share = current_close - stop_price
                
                if risk_per_share > 0:
                    risk_amount = self.risk_pct * self.equity
                    position_size = int(round(risk_amount / risk_per_share))
                    atr_trail = 2 * self.atr[-1]
                    
                    if position_size > 0:
                        self.buy(size=position_size, 
                                sl=stop_price,
                                tp=None,  # No take profit, using trailing stop
                                ts=atr_trail)
                        print(f"🚀🌙 LONG ENTRY! Size: {position_size}")
                        print(f"   SL: {stop_price:.2f} | Trail: {atr_trail:.2f}")
            
            # Short Entry Conditions
            elif ((self.vi_minus[-2] < self.vi_plus[-2] and self.vi_minus[-1] > self.vi_plus[-1]) and \
                 (self.cmo[-1] > 50) and \
                 (self.data.Volume[-1] > self.volume_ma[-1]):
                
                stop_price = self.swing_high[-1]
                risk_per_share = stop_price - current_close
                
                if risk_per_share > 0:
                    risk_amount = self.risk_pct * self.equity
                    position_size = int(round(risk