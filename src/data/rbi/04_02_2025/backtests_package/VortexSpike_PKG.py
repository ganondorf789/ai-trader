I'll fix the code by removing all backtesting.lib imports and replacing the crossover/crossunder functions with manual implementations. Here's the corrected version with Moon Dev themed debug prints:

```python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import talib
import pandas_ta as pta
from backtesting import Strategy, Backtest

class VortexSpike(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    atr_period = 14
    bb_period = 20
    bb_std = 2
    volume_ma_period = 20
    trailing_multiplier = 2
    max_bars_held = 5  # Time-based exit
    
    def init(self):
        # Clean data columns
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])

        # Calculate indicators using TA-Lib/pandas_ta with self.I()
        # Vortex Indicator
        self.vi_plus, self.vi_minus = self.I(pta.vortex,
                                             self.data.High, self.data.Low, self.data.Close,
                                             name='VI', lensig=False)
        
        # ATR
        self.atr = self.I(talib.ATR,
                          self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)
        
        # Bollinger Bands Width
        upper, middle, lower = self.I(talib.BBANDS,
                                      self.data.Close,
                                      timeperiod=self.bb_period,
                                      nbdevup=self.bb_std,
                                      nbdevdn=self.bb_std)
        self.bb_width = (upper - lower) / middle
        
        # Volume MA
        self.volume_ma = self.I(talib.SMA,
                                self.data.Volume,
                                timeperiod=self.volume_ma_period)
        
        # Parabolic SAR
        self.sar = self.I(talib.SAR,
                          self.data.High, self.data.Low,
                          acceleration=0.02, maximum=0.2)
        
        # Track trade parameters
        self.entry_price = None
        self.trailing_stop = None
        self.bars_held = 0

    def next(self):
        price = self.data.Close[-1]
        
        # Moon Dev debug prints 🌙
        print(f"\n🌙 Moon Dev Debug - Bar {len(self.data)-1}")
        print(f"✨ VI+:{self.vi_plus[-1]:.4f} | VI-:{self.vi_minus[-1]:.4f}")
        print(f"📈 ATR:{self.atr[-1]:.2f} | BB Width:{self.bb_width[-1]:.4f}")
        print(f"🔔 SAR:{self.sar[-1]:.2f} | Volume:{self.data.Volume[-1]:.2f} vs MA:{self.volume_ma[-1]:.2f}")

        # Exit conditions
        if self.position.is_long:
            self.bars_held += 1
            self.trailing_stop = max(self.trailing_stop, price - self.trailing_multiplier * self.atr[-1])
            
            if price < self.trailing_stop or price < self.sar[-1] or self.bars_held > self.max_bars_held:
                print(f"🚀🌕 MOON EXIT LONG! Price: {price:.2f} | SAR: {self.sar[-1]:.2f}")
                self.position.close()
                
        elif self.position.is_short:
            self.bars_held += 1
            self.trailing_stop = min(self.trailing_stop, price + self.trailing_multiplier * self.atr[-1])
            
            if price > self.trailing_stop or price > self.sar[-1] or self.bars_held > self.max_bars_held:
                print(f"🌑💫 MOON EXIT SHORT! Price: {price:.2f} | SAR: {self.sar[-1]:.2f}")
                self.position.close()

        # Entry conditions
        if not self.position:
            self.bars_held = 0
            
            # Long entry (replaced crossover with manual check)
            if ((self.vi_plus[-2] < self.vi_minus[-2] and self.vi_plus[-1] > self.vi_minus[-1]) and
                self.atr[-1] > self.bb