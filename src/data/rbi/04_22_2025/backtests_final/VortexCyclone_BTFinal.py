```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

class VortexCycloneStrategy(Strategy):
    risk_per_trade = 0.01
    dpo_period = 20
    vi_period = 14
    max_trade_duration = 10

    def init(self):
        self.vi_plus, self.vi_minus = self.I(talib.VORTEX, self.data.High, self.data.Low, self.data.Close, self.vi_period)
        self.dpo = self.I(self.calculate_dpo, self.data.Close, self.dpo_period)
        self.dpo_std = self.I(talib.STDDEV, self.dpo, 20, nbdev=1)
        
    def calculate_dpo(self, close, period):
        sma = talib.SMA(close, timeperiod=period)
        shift_amount = int(period/2 + 1)
        return close - sma.shift(shift_amount)

    def next(self):
        if len(self.data) < 20:
            return
            
        lower_entry = -1.5 * self.dpo_std[-1]
        upper_entry = 1.5 * self.dpo_std[-1]
        upper_exit = 1.0 * self.dpo_std[-1]
        
        # Exit conditions
        for trade in self.trades:
            if trade.is_long:
                if (self.vi_minus[-2] > self.vi_plus[-2] and self.vi_minus[-1] < self.vi_plus[-1]) or (self.dpo[-2] < upper_exit and self.dpo[-1] > upper_exit):
                    trade.close()
                    print(f"🌙✨ LONG exit at {self.data.Close[-1]} (VI-/DPO exit)")
                elif len(self.data) - trade.entry_bar >= self.max_trade_duration:
                    trade.close()
                    print(f"🌙⏳ LONG exit (max duration)")
            else:
                if (self.vi_plus[-2] > self.vi_minus[-2] and self.vi_plus[-1] < self.vi_minus[-1]) or (-self.dpo[-2] < upper_exit and -self.dpo[-1] > upper_exit):
                    trade.close()
                    print(f"🌙✨ SHORT exit at {self.data.Close[-1]} (VI+/DPO exit)")
                elif len(self.data) - trade.entry_bar >= self.max_trade_duration:
                    trade.close()
                    print(f"🌙⏳ SHORT exit (max duration)")
        
        # Entry conditions
        if not self.position:
            # Long entry checks
            vi_cross_up = (self.vi_plus[-2] < self.vi_minus[-2] and self.vi_plus[-1] > self.vi_minus[-1])
            dpo_cross_above = (self.dpo[-2] < lower_entry and self.dpo[-1] > lower_entry)
            
            # Short entry checks
            vi_cross_down = (self.vi_minus[-2] < self.vi_plus[-2] and self.vi_minus[-1] > self.vi_plus[-1])
            dpo_cross_below = (upper_entry[-2] > self.dpo[-2] and upper_entry[-1] < self.dpo[-1])
            
            if vi_cross_up and dpo_cross_above:
                swing_low = min(self.data.Low[-5:])
                risk_amount = self.equity * self.risk_per_trade
                risk_per_unit = self.data.Close[-1] - swing_low
                if risk_per_unit > 0:
                    size = int(round(risk_amount / risk_per_unit))
                    self.buy(size=size, sl=swing_low)
                    print(f"🌪️🚀 LONG ENTRY {size} units at {self.data.Close[-1]}!")
            
            elif vi_cross_down and dpo_cross_below:
                swing_high = max(self.data.High[-5:])
                risk_amount = self.equity * self.risk_per_trade
                risk_per_unit = swing_high - self.data.Close[-1]
                if risk_per_unit > 0:
                    size = int(round(risk_amount / risk_per_unit))
                    self.sell(size=size, sl=swing_high)
                    print(f"🌪️💥 SHORT ENTRY {size} units at {self.data.Close[-1]}!")

# Data handling
data = pd.read_csv('your_data.csv', parse_dates=['Date'], index_col='Date')

# Run