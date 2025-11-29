Here's the fixed code with all `backtesting.lib` references removed and replaced with proper Moon Dev themed implementations:

```python
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

class DivergenceSqueeze(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    atr_multiplier = 1.5
    
    def init(self):
        # Calculate indicators with TA-Lib using self.I()
        self.macd, self.macd_signal, _ = self.I(talib.MACD, self.data.Close, 
                                               fastperiod=12, slowperiod=26, signalperiod=9,
                                               name=['MACD', 'Signal', '_Hist'])
        
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(talib.BBANDS, self.data.Close,
                                                           timeperiod=20, nbdevup=2, nbdevdn=2,
                                                           name=['BB Upper', 'BB Mid', 'BB Lower'])
        
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                         timeperiod=14, name='ATR')
        
        self.vol = self.I(lambda: self.data.Volume, name='Volume')
        
    def next(self):
        price = self.data.Close[-1]
        print(f"🌙 Current Price: {price:.2f} | MACD: {self.macd[-1]:.2f} | Volume: {self.vol[-1]:.2f}")
        
        if not self.position:
            # Bullish divergence check
            if (self.data.Low[-1] < self.data.Low[-2] and
                self.macd[-1] > self.macd[-2] and
                self.vol[-1] < self.vol[-2]):
                
                self.enter_long()
            
            # Bearish divergence check
            elif (self.data.High[-1] > self.data.High[-2] and
                  self.macd[-1] < self.macd[-2] and
                  self.vol[-1] < self.vol[-2]):
                
                self.enter_short()
        else:
            self.check_exits()
    
    def enter_long(self):
        price = self.data.Close[-1]
        risk_amount = self.equity * self.risk_pct
        atr_val = self.atr[-1] or 0.01
        stop_loss = price - (self.atr_multiplier * atr_val)
        position_size = int(round(risk_amount / (self.atr_multiplier * atr_val)))
        
        if position_size > 0:
            self.buy(size=position_size, sl=stop_loss)
            print(f"🚀🌕 MOON DEV LONG ENTRY! Size: {position_size} | Entry: {price:.2f} | SL: {stop_loss:.2f}")
    
    def enter_short(self):
        price = self.data.Close[-1]
        risk_amount = self.equity * self.risk_pct
        atr_val = self.atr[-1] or 0.01
        stop_loss = price + (self.atr_multiplier * atr_val)
        position_size = int(round(risk_amount / (self.atr_multiplier * atr_val)))
        
        if position_size > 0:
            self.sell(size=position_size, sl=stop_loss)
            print(f"🌑🌙 MOON DEV SHORT ENTRY! Size: {position_size} | Entry: {price:.2f} | SL: {stop_loss:.2f}")
    
    def check_exits(self):
        if self.position.is_long:
            # Bullish crossover detection without backtesting.lib
            if (self.data.High[-1] >= self.bb_upper[-1] or
                (self.macd_signal[-2] > self.macd[-2] and self.macd_signal[-1] < self.macd[-1])):
                self.position.close()
                print(f"✨🎯 Closing LONG at {self.data.Close[-1]:.2f} (Exit Signal)")
        
        elif self.position.is_short:
            # Bearish crossover detection without backtesting.lib
            if (self.data.Low[-1] <= self.bb_lower[-1] or
                (self.macd[-2] > self.macd_signal[-2] and self.macd[-1] < self.macd_signal[-1])):
                self.position.close()
                print(f"✨🎯 Closing SHORT at {self.data.Close[-1]:.2