I'll fix the code while maintaining the original strategy logic. Here's the corrected version with Moon Dev themed debug prints:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

class VoltaVwap(Strategy):
    def init(self):
        # Clean and prepare data - Moon Dev data sanitization 🌙
        if hasattr(self.data, 'df'):
            self.data.df.columns = self.data.df.columns.str.strip().str.lower()
            self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns 
                                                    if 'unnamed' in str(col).lower()])
            print("🌙✨ Moon Dev cleaned data columns")

        # Calculate volatility-adjusted VWAP
        typical_price = (self.data.High + self.data.Low + self.data.Close) / 3
        numerator = (typical_price * self.data.Volume).rolling(20).sum()
        denominator = self.data.Volume.rolling(20).sum()
        vwap = numerator / denominator
        std_dev = self.data.Close.rolling(20).std()
        atr = talib.ATR(self.data.High, self.data.Low, self.data.Close, 14)
        self.volatility_vwap = vwap + (std_dev * atr)
        print("🌙✨ Moon Dev calculated Volatility VWAP")

        # Calculate Bollinger Bands
        self.upper, self.middle, self.lower = talib.BBANDS(self.data.Close, 20, 2, 2)
        bb_width = (self.upper - self.lower) / self.middle
        self.bb_width_min = bb_width.rolling(960).min()  # 10 days in 15m
        print("🌙✨ Moon Dev calculated Bollinger Bands")

        # Calculate ADR with daily resampling
        daily_high = self.data.High.resample('D').max().ffill()
        daily_low = self.data.Low.resample('D').min().ffill()
        daily_range = daily_high - daily_low
        self.adr = daily_range.rolling(14).mean()
        print("🌙✨ Moon Dev calculated ADR")

        # Track trading session
        self.end_of_day = (self.data.index.to_series().diff(-1).dt.days > 0)
        print("🌙✨ Moon Dev session tracking initialized")

    def next(self):
        # Moon Dev position sizing - 1% risk per trade 🌙
        risk_per_trade = 0.01
        position_size = risk_per_trade * self.equity / (self.data.Close[-1] * 0.01)  # Assuming 1% stop
        position_size = int(position_size)  # Convert to whole units
        
        # Entry conditions
        long_entry = (crossover(self.data.Close, self.volatility_vwap) and 
                     (self.bb_width[-1] <= self.bb_width_min[-1]))
        
        short_entry = (crossunder(self.data.Close, self.volatility_vwap) and 
                      (self.bb_width[-1] <= self.bb_width_min[-1]))
        
        # Moon Dev trade management 🌙
        if not self.position:
            if long_entry:
                self.stop_loss = self.volatility_vwap[-1]
                self.take_profit = self.data.Close[-1] + (2 * self.adr[-1])
                self.adr_at_entry = self.adr[-1]
                self.buy(size=position_size)
                print(f"🌙🚀 Moon Dev LONG entered @ {self.data.Close[-1]} "
                      f"| TP: {self.take_profit:.2f} | SL: {self.stop_loss:.2f}")
            
            elif short_entry:
                self.stop_loss = self.volatility_vwap[-1]
                self.take_profit = self.data.Close[-1] - (2 * self.adr[-1])
                self.adr_at_entry = self.adr[-1]
                self.sell(size=position_size)
                print(f"🌙🚀 Moon Dev SHORT entered @ {self.data.Close[-1]} "
                      f"| TP: {self.take_profit:.2f} | SL: {self.stop_loss:.2f}")
        
        else:
            # Moon Dev exit conditions 🌙
            if self.position.is_long:
                # Update trailing stop
                if self.data.High[-1] >= self.entry_price + self.adr_at_entry:
                    self.stop_loss = self.entry_price
                    print(f"✨ Moon Dev Trailing stop moved to breakeven @ {self.stop_loss:.2f}")
                
                #