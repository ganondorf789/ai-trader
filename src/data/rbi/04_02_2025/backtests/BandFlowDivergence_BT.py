```python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class BandFlowDivergence(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    time_exit_bars = 5  # Bars to hold before time exit
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators with self.I()
        # Bollinger Bands (20, 2)
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name='BB_UPPER')
        self.bb_middle = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name='BB_MIDDLE')
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name='BB_LOWER')
        
        # Money Flow Index (14)
        self.mfi = self.I(talib.MFI, self.data.High, self.data.Low, self.data.Close, self.data.Volume, 14)
        
        # Keltner Channels (20, 1.5 ATR)
        def calculate_kc():
            kc = ta.kc(self.data.High, self.data.Low, self.data.Close, 20, 1.5)
            return kc.iloc[:,0], kc.iloc[:,1], kc.iloc[:,2]
        self.kc_upper, self.kc_middle, self.kc_lower = self.I(calculate_kc, name='KC_UPPER'), self.I(calculate_kc, name='KC_MIDDLE'), self.I(calculate_kc, name='KC_LOWER')
        
        # Swing high/low using MAX/MIN
        self.swing_high = self.I(talib.MAX, self.data.High, 20)
        self.swing_low = self.I(talib.MIN, self.data.Low, 20)
        
        # Volume SMA for filter
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20)
        
        print("🌙✨ Strategy initialized with Moon Dev precision!")

    def next(self):
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        
        # Entry Logic
        if not self.position:
            # Long Entry Conditions
            long_cond = (current_close > self.bb_upper[-1] and
                        current_high > self.swing_high[-2] and
                        self.mfi[-1] < self.mfi[-2] and
                        self.data.Volume[-1] > self.volume_sma[-1])
            
            # Short Entry Conditions
            short_cond = (current_close < self.bb_lower[-1] and
                         current_low < self.swing_low[-2] and
                         self.mfi[-1] > self.mfi[-2] and
                         self.data.Volume[-1] > self.volume_sma[-1])
            
            # Execute entries with risk management
            if long_cond:
                self.enter_long()
            elif short_cond:
                self.enter_short()
                
        # Exit Logic
        else:
            if self.position.is_long:
                if current_close < self.kc_upper[-1]:
                    self.position.close()
                    print(f"✨🌙 EXIT LONG: Price closed inside KC Upper at {current_close:.2f}")
            elif self.position.is_short:
                if current_close > self.kc_lower[-1]:
                    self.position.close()
                    print(f"✨🌙 EXIT SHORT: Price closed inside KC Lower at {current_close:.2f}")
            
            # Time-based exit
            if len(self.data) - self.position.entry_bar >= self.time_exit_bars:
                self.position.close()
                print(f"⏳🌙 TIME EXIT: Held for {self.time_exit_bars} bars")

    def enter_long(self):
        bb_mid = (self.bb_upper[-1] + self.bb_lower