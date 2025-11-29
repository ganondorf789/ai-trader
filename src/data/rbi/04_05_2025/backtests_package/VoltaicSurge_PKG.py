Here's the fixed and completed code with all Moon Dev themed improvements and proper indicator handling:

```python
import pandas as pd
import talib as ta
import numpy as np
from backtesting import Strategy, Backtest

class VoltaicSurge(Strategy):
    risk_percent = 0.01
    bb_period = 20
    bb_dev = 2
    adx_period = 14
    atr_period = 14
    bb_lookback = 100
    bb_percentile = 20

    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col], inplace=True)

        # Calculate indicators with TA-Lib
        self.upper_bb, self.middle_bb, self.lower_bb = self.I(ta.BBANDS, self.data.Close, 
                                                            timeperiod=self.bb_period, 
                                                            nbdevup=self.bb_dev, 
                                                            nbdevdn=self.bb_dev, 
                                                            matype=0,
                                                            name=['UpperBB', 'MiddleBB', 'LowerBB'])
        
        # Calculate Bollinger Band Width
        self.bb_width = self.I(lambda u, l, m: (u - l)/m, 
                             self.upper_bb, self.lower_bb, self.middle_bb, 
                             name='BB_Width')
        
        # Calculate BB Width Percentile
        self.bb_width_percentile = self.I(lambda x: x.rolling(self.bb_lookback).quantile(self.bb_percentile/100),
                                        self.bb_width, name='BB_Width_Pctl')
        
        # Calculate ADX
        self.adx = self.I(ta.ADX, self.data.High, self.data.Low, self.data.Close, 
                        timeperiod=self.adx_period, name='ADX')
        
        # Calculate ATR
        self.atr = self.I(ta.ATR, self.data.High, self.data.Low, self.data.Close,
                        timeperiod=self.atr_period, name='ATR')
        
        # Track trade parameters
        self.entry_price = None
        self.initial_stop = None
        self.highest_high = None
        self.lowest_low = None

    def next(self):
        price = self.data.Close[-1]
        adx = self.adx[-1]
        bb_width = self.bb_width[-1]
        bb_pctl = self.bb_width_percentile[-1]

        # Moon Dev debug prints 🌙
        print(f"🌙 Current Price: {price:.2f} | ADX: {adx:.1f} | BB%: {bb_width*100:.2f}%/{bb_pctl*100:.2f}% ✨")

        if not self.position:
            # Long entry conditions
            if (bb_width < bb_pctl and
                adx > 25 and
                price > self.upper_bb[-1]):
                
                atr_value = self.atr[-1]
                self.entry_price = price
                self.initial_stop = self.entry_price - 1.5 * atr_value
                self.highest_high = self.data.High[-1]
                
                # Calculate position size
                risk_amount = self.equity * self.risk_percent
                risk_per_share = 1.5 * atr_value
                position_size = int(round(risk_amount / risk_per_share))
                
                self.buy(size=position_size)
                print(f"🌙🚀 LONG ENTRY @ {price:.2f} | Size: {position_size} | SL: {self.initial_stop:.2f} ✨")

            # Short entry conditions    
            elif (bb_width < bb_pctl and
                  adx > 25 and
                  price < self.lower_bb[-1]):
                
                atr_value = self.atr[-1]
                self.entry_price = price
                self.initial_stop = self.entry_price + 1.5 * atr_value
                self.lowest_low = self.data.Low[-1]
                
                risk_amount = self.equity * self.risk_percent
                risk_per_share = 1.5 * atr_value
                position_size = int(round(risk_amount / risk_per_share))
                
                self.sell(size=position_size)
                print(f"🌙🚀 SHORT ENTRY @ {price:.2f} | Size: {position_size} | SL: {self.initial_stop:.2f