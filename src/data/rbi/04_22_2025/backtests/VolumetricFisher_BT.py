```python
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

class VolumetricFisher(Strategy):
    def init(self):
        # Data handling
        data = self.data.df
        data.columns = data.columns.str.strip().str.lower()
        data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

        # Indicators
        # Fisher Transform with Signal
        self.fisher, self.fisher_signal = self.I(ta.fisher, self.data.High, self.data.Low, length=10, signal=9, name='Fisher')
        
        # Volume-Weighted MACD
        vw_close = self.data.Close * self.data.Volume
        self.macd_line = self.I(talib.MACD, vw_close, fastperiod=12, slowperiod=26, signalperiod=9, name='VW_MACD')[0]
        self.macd_signal = self.I(talib.MACD, vw_close, fastperiod=12, slowperiod=26, signalperiod=9, name='VW_MACD_Signal')[1]
        
        # Standard Deviation Bands
        self.fisher_mean = self.I(talib.SMA, self.fisher, timeperiod=14, name='Fisher_Mean')
        self.fisher_std = self.I(talib.STDDEV, self.fisher, timeperiod=14, nbdev=1, name='Fisher_Std')
        self.macd_mean = self.I(talib.SMA, self.macd_line, timeperiod=14, name='MACD_Mean')
        self.macd_std = self.I(talib.STDDEV, self.macd_line, timeperiod=14, nbdev=1, name='MACD_Std')
        
        # ATR for Risk Management
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR')

    def next(self):
        # Entry/Exit Conditions
        fisher_above_upper = self.fisher[-1] > (self.fisher_mean[-1] + 2*self.fisher_std[-1])
        macd_above_upper = self.macd_line[-1] > (self.macd_mean[-1] + 2*self.macd_std[-1])
        fisher_below_lower = self.fisher[-1] < (self.fisher_mean[-1] - 2*self.fisher_std[-1])
        macd_below_lower = self.macd_line[-1] < (self.macd_mean[-1] - 2*self.macd_std[-1])
        
        long_entry = (crossover(self.fisher, self.fisher_signal) and 
                     crossover(self.macd_line, self.macd_signal) and 
                     fisher_above_upper and macd_above_upper)
        
        short_entry = (crossunder(self.fisher, self.fisher_signal) and 
                      crossunder(self.macd_line, self.macd_signal) and 
                      fisher_below_lower and macd_below_lower)
        
        # Risk Management
        risk_percent = 0.01
        atr_value = self.atr[-1]
        
        # Position Sizing
        if long_entry and not self.position:
            sl_price = self.data.Close[-1] - 1.5 * atr_value
            risk_amount = self.equity * risk_percent
            position_size = int(round(risk_amount / (self.data.Close[-1] - sl_price)))
            if position_size > 0:
                self.buy(size=position_size, sl=sl_price)
                print(f"🌙 MOON DEV LONG ENTRY 🚀 | Size: {position_size} | Price: {self.data.Close[-1]:.2f} | SL: {sl_price:.2f}")
        
        elif short_entry and not self.position:
            sl_price = self.data.Close[-1] + 1.5 * atr_value
            risk_amount = self.equity * risk_percent
            position_size = int(round(risk_amount / (sl_price - self.data.Close[-1])))
            if position_size > 0:
                self.sell(size=position_size, sl=sl_price)
                print(f"🌙 MOON DEV SHORT ENTRY 🌪️ | Size: {position_size} | Price: {self.data.Close[-1]:.2f} | SL: {sl