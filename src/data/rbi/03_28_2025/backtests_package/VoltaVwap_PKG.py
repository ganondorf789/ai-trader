I'll fix the code by removing all `backtesting.lib` imports and replacing them with proper alternatives. Here's the corrected version with Moon Dev themed debug prints:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

class VoltaVwap(Strategy):
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate volatility-adjusted VWAP
        typical_price = (self.data.High + self.data.Low + self.data.Close) / 3
        numerator = self.I(talib.SUM, typical_price * self.data.Volume, 20)
        denominator = self.I(talib.SUM, self.data.Volume, 20)
        vwap = numerator / denominator
        std_dev = self.I(talib.STDDEV, self.data.Close, 20)
        atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.volatility_vwap = self.I(lambda: vwap + (std_dev * atr), name='Volatility VWAP')

        # Calculate Bollinger Bands
        self.middle, self.upper, self.lower = self.I(talib.BBANDS, self.data.Close, 20, 2, 2)
        bb_width = (self.upper - self.lower) / self.middle
        self.bb_width_min = self.I(talib.MIN, bb_width, 960)  # 10 days in 15m

        # Calculate ADR with daily resampling
        # Moon Dev custom resampling implementation 🌙
        daily_high = self.data.High.resample('D').max().reindex(self.data.High.index, method='ffill')
        daily_low = self.data.Low.resample('D').min().reindex(self.data.Low.index, method='ffill')
        daily_range = daily_high - daily_low
        self.adr = self.I(talib.SMA, daily_range, 14, name='ADR')

        # Track trading session
        self.end_of_day = self.I(lambda: self.data.df.index.to_series().diff(-1).dt.days > 0, name='EOD')

    def next(self):
        # Moon Dev risk management checks 🌙
        if self.position:
            if self.position.is_long:
                # Update trailing stop
                if self.data.High[-1] >= self.entry_price + self.adr_at_entry:
                    self.stop_loss = self.entry_price
                    print(f"✨ Moon Dev Trailing stop moved to breakeven @ {self.stop_loss}")
                
                # Check exits
                if self.data.Low[-1] <= self.stop_loss:
                    self.position.close()
                    print(f"🌙🛑 Moon Dev Long stopped out @ {self.stop_loss}")
                elif self.data.High[-1] >= self.take_profit:
                    self.position.close()
                    print(f"🚀✅ Moon Dev Long TP hit @ {self.take_profit}")
                elif (self.volatility_vwap[-2] < self.data.Close[-2] and 
                      self.volatility_vwap[-1] > self.data.Close[-1]):
                    self.position.close()
                    print(f"🌙✨ Moon Dev Opposing signal exit @ {self.data.Close[-1]}")
            
            elif self.position.is_short:
                # Update trailing stop
                if self.data.Low[-1] <= self.entry_price - self.adr_at_entry:
                    self.stop_loss = self.entry_price
                    print(f"✨ Moon Dev Trailing stop moved to breakeven @ {self.stop_loss}")
                
                # Check exits
                if self.data.High[-1] >= self.stop_loss:
                    self.position.close()
                    print(f"🌙🛑 Moon Dev Short stopped out @ {self.stop_loss}")
                elif self.data.Low[-1] <= self.take_profit:
                    self.position.close()
                    print(f"🚀✅ Moon Dev Short TP hit @ {self.take_profit}")
                elif (self.data.Close[-2] < self.volatility_vwap[-2] and 
                      self.data.Close[-1] > self.volatility_vwap[-1]):
                    self.position.close()
                    print(f"🌙✨ Moon Dev Opposing signal exit @ {self.data.Close[-1]}")
            
            # End-of-day close
            if self.end_of_day[-1]:
                self.position.close()
                print(f"🌙⌛ Moon Dev EOD position