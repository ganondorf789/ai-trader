```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class VoltaSqueeze(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # Clean data and prepare columns
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators using TA-Lib with self.I() wrapper
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume
        
        # Bollinger Bands (20,2)
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        self.upper_band = self.I(lambda: upper, name='UPPER')
        self.lower_band = self.I(lambda: lower, name='LOWER')
        
        # Bollinger Band Width Percentile
        bb_width = (upper - lower) / middle
        self.bbwp = self.I(ta.percentrank, bb_width, length=100, name='BBWP')
        
        # On-Balance Volume
        self.obv = self.I(talib.OBV, close, volume, name='OBV')
        
        # Swing detection (5-period lookback)
        self.high_swings = self.I(talib.MAX, high, timeperiod=5, name='SWING_HIGH')
        self.low_swings = self.I(talib.MIN, low, timeperiod=5, name='SWING_LOW')
        self.obv_high = self.I(talib.MAX, self.obv, timeperiod=5, name='OBV_HIGH')
        self.obv_low = self.I(talib.MIN, self.obv, timeperiod=5, name='OBV_LOW')
        
    def next(self):
        # Wait for sufficient data
        if len(self.data) < 100:
            return
        
        current_close = self.data.Close[-1]
        bbwp = self.bbwp[-1]
        obv = self.obv[-1]
        
        # Moon Dev Divergence Detection 🌛✨
        price_bull_div = (self.low_swings[-1] < self.low_swings[-2] and 
                         self.obv_low[-1] > self.obv_low[-2])
        
        price_bear_div = (self.high_swings[-1] > self.high_swings[-2] and 
                         self.obv_high[-1] < self.obv_high[-2])
        
        # Entry Logic with Moon Dev Flair 🌙🚀
        if not self.position:
            # Long Entry
            if (bbwp < 20 and 
                current_close > self.upper_band[-1] and 
                price_bull_div):
                
                sl = self.lower_band[-1]
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = abs(current_close - sl)
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=sl)
                    print(f"🌙 MOON DEV LONG SIGNAL 🌛 | Price: {current_close:.2f} | BBWP: {bbwp:.1f} | Size: {position_size} 🚀")
            
            # Short Entry
            elif (bbwp < 20 and 
                  current_close < self.lower_band[-1] and 
                  price_bear_div):
                
                sl = self.upper_band[-1]
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = abs(sl - current_close)
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    self.sell(size=position_size, sl=sl)
                    print(f"🌙 MOON DEV SHORT SIGNAL 🌜 | Price: {current_close:.2f} | BBWP: {bbwp:.1f} | Size: {position_size} 🚀")

        # Exit Logic with Cosmic Precision 🌌
        elif self.position.is_long and current_close < self.upper_band[-1]:
            self.position.close()