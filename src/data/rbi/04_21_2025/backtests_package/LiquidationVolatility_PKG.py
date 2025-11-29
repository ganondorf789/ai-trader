Here's the fixed code with all backtesting.lib references removed and proper indicator implementations:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Load and preprocess data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class LiquidationVolatility(Strategy):
    def init(self):
        # Core indicators
        self.upper, self.middle, self.lower = self.I(talib.BBANDS, self.data.Close, 20, 2, 2, name='BBANDS')
        self.bbw = self.I(lambda: (self.upper - self.lower) / self.middle, name='BBW')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        self.swing_high = self.I(talib.MAX, self.data.High, 50, name='SwingHigh')
        self.swing_low = self.I(talib.MIN, self.data.Low, 50, name='SwingLow')
        self.highest_high = self.I(talib.MAX, self.data.High, 22, name='HighestHigh')
        self.lowest_low = self.I(talib.MIN, self.data.Low, 22, name='LowestLow')

    def next(self):
        if self.position:
            self.trailing_stop()
            self.volatility_exit()
        else:
            self.check_entries()

    def trailing_stop(self):
        if self.position.is_long:
            trail = self.highest_high[-1] - 3*self.atr[-1]
            self.position.sl = max(self.position.sl or 0, trail)
        else:
            trail = self.lowest_low[-1] + 3*self.atr[-1]
            self.position.sl = min(self.position.sl or float('inf'), trail)

    def volatility_exit(self):
        bbw_window = self.bbw[-35040:] if len(self.bbw) > 35040 else self.bbw
        if len(bbw_window) < 100: return
        
        current_bbw = self.bbw[-1]
        if current_bbw > np.percentile(bbw_window, 75):
            self.position.close()
            print(f"🌪️🌙 Volatility spike! Closing at {self.data.Close[-1]:.2f} (BBW: {current_bbw:.2f})")

    def check_entries(self):
        # BBW percentile check
        bbw_window = self.bbw[-35040:] if len(self.bbw) > 35040 else self.bbw
        if len(bbw_window) < 100: return
        
        bbw_p20 = np.percentile(bbw_window, 20)
        bbw_p50 = np.percentile(bbw_window, 50)
        current_bbw = self.bbw[-1]
        
        if current_bbw > bbw_p50 or current_bbw > bbw_p20:
            return

        # Price near liquidation levels
        price = self.data.Close[-1]
        near_high = price >= self.swing_high[-1]*0.98
        near_low = price <= self.swing_low[-1]*1.02
        
        if not (near_high or near_low):
            return

        # Position sizing
        risk_amount = self.equity * 0.01
        atr_value = self.atr[-1]
        
        if near_high and price > self.swing_high[-1]:
            sl_price = price - 1.5*atr_value
            size = int(round(risk_amount / (price - sl_price)))
            if size > 0:
                self.buy(size=size, sl=sl_price)
                print(f"🚀🌙 LONG! {size} units @ {price:.2f} | SL: {sl_price:.2f}")
                
        elif near_low and price < self.swing_low[-