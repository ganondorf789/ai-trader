I'll help you fix the code by removing the `backtesting.lib` import and ensuring proper indicator usage. Here's the corrected version:

```python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class DeltaBreakout(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators using TA-Lib
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=20)
        
        # Calculate Volume Delta
        close_prices = self.data.Close.values
        open_prices = self.data.Open.values
        volumes = self.data.Volume.values
        self.volume_delta = self.I(lambda: np.where(close_prices > open_prices, volumes,
                                                   np.where(close_prices < open_prices, -volumes, 0)),
                                  name='Volume_Delta')
        
        # Volume Delta moving average
        self.volume_delta_ma = self.I(talib.SMA, self.volume_delta, timeperiod=20)
        
        # ATR moving average for consolidation detection
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=20)
        
        print("🌙✨ Moon Dev Indicators Initialized! Ready for Launch! 🚀")

    def next(self):
        current_close = self.data.Close[-1]
        current_ema = self.ema[-1]
        current_atr = self.atr[-1]
        upper_kc = current_ema + 2 * current_atr
        lower_kc = current_ema - 2 * current_atr
        
        # Volume analysis
        vol_delta = self.volume_delta[-1]
        vol_delta_ma = self.volume_delta_ma[-1]
        atr_ma = self.atr_ma[-1]
        
        if not self.position:
            # Check consolidation (ATR below its MA)
            consolidation = current_atr < atr_ma
            
            # Long entry conditions
            if (current_close > upper_kc and 
                vol_delta > 2 * vol_delta_ma and 
                consolidation):
                
                risk_amount = self.equity * self.risk_percent
                risk_per_share = current_close - lower_kc
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.buy(size=position_size, sl=lower_kc)
                        print(f"🌙✨🚀 MOON DEV LONG SIGNAL! Buying {position_size} units at {current_close:.2f}")
            
            # Short entry conditions
            elif (current_close < lower_kc and 
                  vol_delta < -2 * vol_delta_ma and 
                  consolidation):
                
                risk_amount = self.equity * self.risk_percent
                risk_per_share = upper_kc - current_close
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.sell(size=position_size, sl=upper_kc)
                        print(f"🌙✨🚀 MOON DEV SHORT SIGNAL! Selling {position_size} units at {current_close:.2f}")
        
        else:
            # Exit conditions for long
            if self.position.is_long:
                if vol_delta < 0 or current_close < lower_kc:
                    self.position.close()
                    print(f"🌙✨ MOON DEV LONG EXIT! Closed at {current_close:.2f} 🌌")
            
            # Exit conditions for short
            elif self.position.is_short:
                if vol_delta > 0 or current_close > upper_kc:
                    self.position.close()
                    print(f"🌙✨ MOON DEV SHORT EXIT! Closed at {current_close:.2f} 🌌")

# Data preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean data
data.columns = data.columns.str.strip().str.lower()
data =