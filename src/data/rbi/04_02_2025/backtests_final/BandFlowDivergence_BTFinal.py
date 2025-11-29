I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete fixed version with Moon Dev themed debug prints:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

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
        
        print("🌙✨ Moon Dev Package AI initialized with celestial precision!")
        print("🌌 Indicators activated: Bollinger Bands | MFI | Keltner Channels | Swing Points")

    def enter_long(self):
        stop_price = self.bb_middle[-1]  # Stop at BB middle band
        risk_amount = self.equity * self.risk_per_trade
        risk_per_share = self.data.Close[-1] - stop_price
        position_size = int(risk_amount / risk_per_share)
        
        if position_size > 0:
            self.buy(size=position_size, sl=stop_price)
            print(f"🌙✨ ENTER LONG: {position_size} units at {self.data.Close[-1]:.2f} | SL: {stop_price:.2f}")

    def enter_short(self):
        stop_price = self.bb_middle[-1]  # Stop at BB middle band
        risk_amount = self.equity * self.risk_per_trade
        risk_per_share = stop_price - self.data.Close[-1]
        position_size = int(risk_amount / risk_per_share)
        
        if position_size > 0:
            self.sell(size=position_size, sl=stop_price)
            print(f"🌙✨ ENTER SHORT: {position_size} units at {self.data.Close[-1]:.2f} | SL: {stop_price:.2f}")

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
                        self.data.