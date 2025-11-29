Here's the fixed code with all backtesting.lib imports removed and proper Moon Dev themed debug prints added:

```python
import pandas as pd
import talib as ta
from backtesting import Backtest, Strategy

class VortexDivergence(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    lookback_div = 5
    swing_period = 20
    
    def init(self):
        # Calculate indicators with self.I()
        self.vi_plus = self.I(lambda h, l, c: ta.VORTEX(h, l, c, 14)[0], 
                           self.data.High, self.data.Low, self.data.Close)
        self.vi_minus = self.I(lambda h, l, c: ta.VORTEX(h, l, c, 14)[1], 
                            self.data.High, self.data.Low, self.data.Close)
        self.mfi = self.I(ta.MFI, self.data.High, self.data.Low, 
                        self.data.Close, self.data.Volume, 14)
        self.adx = self.I(ta.ADX, self.data.High, self.data.Low, 
                        self.data.Close, 14)
        self.sma200 = self.I(ta.SMA, self.data.Close, 200)
        self.swing_low = self.I(ta.MIN, self.data.Low, self.swing_period)
        self.swing_high = self.I(ta.MAX, self.data.High, self.swing_period)
        
        # Track trade parameters
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.trailing_trigger = None

    def next(self):
        price = self.data.Close[-1]
        adx = self.adx[-1]
        
        # Moon Dev debug prints 🌙✨
        print(f"🌙 Price: {price:.2f} | VI+:{self.vi_plus[-1]:.2f} VI-:{self.vi_minus[-1]:.2f} | ADX: {adx:.2f}")
        
        # Exit conditions
        if self.position:
            if adx < 20:
                print(f"🌘 ADX dropped below 20! Closing position!")
                self.position.close()
            elif self.position.is_long and price >= self.take_profit:
                print(f"🚀✨ Take profit reached! Long closed at {price:.2f}")
                self.position.close()
            elif self.position.is_short and price <= self.take_profit:
                print(f"🚀✨ Take profit reached! Short closed at {price:.2f}")
                self.position.close()
            elif self.position.is_long and price >= self.trailing_trigger:
                print(f"🛡️ Moving stop to breakeven @ {self.entry_price:.2f}")
                self.stop_loss = self.entry_price
                self.trailing_trigger = None  # Disable further trailing
            return
        
        # Long entry conditions - replaced crossover with manual check
        if ((self.vi_plus[-2] < self.vi_minus[-2] and self.vi_plus[-1] > self.vi_minus[-1]) and
            price > self.sma200[-1] and
            adx > 25):
            
            # Bullish divergence check
            if len(self.mfi) > self.lookback_div:
                price_low = price < self.data.Close[-self.lookback_div]
                mfi_high = self.mfi[-1] > self.mfi[-self.lookback_div]
                if price_low and mfi_high:
                    self._enter_trade(direction='long')

        # Short entry conditions - replaced crossover with manual check        
        elif ((self.vi_minus[-2] > self.vi_plus[-2] and self.vi_minus[-1] < self.vi_plus[-1]) and
              price < self.sma200[-1] and
              adx > 25):
            
            # Bearish divergence check
            if len(self.mfi) > self.lookback_div:
                price_high = price > self.data.Close[-self.lookback_div]
                mfi_low = self.mfi[-1] < self.mfi[-self.lookback_div]
                if price_high and mfi_low:
                    self._enter_trade(direction='short')

    def _enter_trade(self, direction):
        price = self.data.Close[-1]
        risk_amount = self.broker.equity * self.risk_per_trade
        
        if direction == 'long':
            sl = self.swing_low[-1]
            risk_per_unit = price - sl