Here's the fully debugged and Moon Dev themed version of your SqueezeExpansion strategy:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class SqueezeExpansion(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    fib_level = 1.618
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col], inplace=True)

        # Calculate indicators using TA-Lib
        # Bollinger Bands (20,2)
        self.upper = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0]
        self.middle = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1]
        self.lower = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2]
        
        # Bollinger Band Width (BBW)
        self.bbw = self.I(lambda: (self.upper - self.lower) / self.middle)
        
        # BBW Squeeze Detection
        self.bbw_low = self.I(talib.MIN, self.bbw, timeperiod=20)
        
        # Accumulation/Distribution Line
        self.adl = self.I(talib.AD, self.data.High, self.data.Low, self.data.Close, self.data.Volume)
        
        # ADL Momentum (2-period ROC)
        self.adl_roc = self.I(talib.ROC, self.adl, timeperiod=2)
        
        # Volatility Stop (14-period ATR)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)

    def next(self):
        price = self.data.Close[-1]
        equity = self.equity
        
        # Entry Logic 🌙
        if not self.position:
            # Squeeze condition
            squeeze_active = self.bbw[-1] <= self.bbw_low[-1]
            
            # Long Entry
            if (squeeze_active and
                self.adl_roc[-1] > 0 and
                price > self.upper[-1]):
                
                stop_loss = price - 1.5 * self.atr[-1]
                risk_amount = equity * self.risk_per_trade
                position_size = risk_amount / (price - stop_loss)
                position_size = int(round(position_size))  # Ensure whole units
                
                if position_size > 0:
                    squeeze_range = self.upper[-1] - self.lower[-1]
                    take_profit = price + squeeze_range * self.fib_level
                    self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                    print(f"🌙🚀 MOON DEV LONG SIGNAL! Buying {position_size} units at {price:.2f}")
                    print(f"🌕✨ Stop: {stop_loss:.2f} | Target: {take_profit:.2f} | R/R: {((take_profit-price)/(price-stop_loss)):.2f}:1")
                    print(f"🌌 Squeeze Range: {squeeze_range:.2f} | ATR: {self.atr[-1]:.2f}")

            # Short Entry    
            elif (squeeze_active and
                  self.adl_roc[-1] < 0 and
                  price < self.lower[-1]):
                  
                stop_loss = price + 1.5 * self.atr[-1]
                risk_amount = equity * self.risk_per_trade
                position_size = risk_amount / (stop_loss - price)
                position_size = int(round(position_size))  # Ensure whole units
                
                if position_size > 0:
                    squeeze_range = self.upper[-1] - self.lower[-1]
                    take_profit = price - squeeze_range * self.fib_level
                    self.sell(size=position_size, sl=stop_loss, tp=take_profit)
                    print(f"🌙🌒 MOON DEV SHORT SIGNAL! Selling {position_size} units at {price:.2f}")
                    print(f"🌑💫 Stop: {stop_loss:.2f} | Target: {take_pro