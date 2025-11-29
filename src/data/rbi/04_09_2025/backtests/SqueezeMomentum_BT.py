Here's the complete implementation of the SqueezeMomentum strategy for backtesting.py:

```python
# 🌙 Moon Dev Backtest AI Generated Code 🚀

# 1. Necessary imports
from backtesting import Backtest, Strategy
import pandas as pd
import talib

# 2. Data handling
def prepare_data(filepath):
    # Load and clean data
    data = pd.read_csv(filepath)
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    
    # Rename and format columns
    data.rename(columns={
        'datetime': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

# 3. Strategy class
class SqueezeMomentum(Strategy):
    max_consecutive_losses = 3
    risk_per_trade = 0.02
    
    def init(self):
        # 🌙 Calculate indicators using TA-Lib
        # Bollinger Bands
        self.upper_band = self.I(lambda c: talib.BBANDS(c, 20, 2, 2, 0)[0], self.data.Close)
        self.lower_band = self.I(lambda c: talib.BBANDS(c, 20, 2, 2, 0)[2], self.data.Close)
        
        # OBV indicators
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume)
        self.obv_ma = self.I(talib.SMA, self.obv, 5)
        self.obv_high = self.I(talib.MAX, self.obv, 10)
        
        # Track trading state
        self.consecutive_losses = 0
        print("✨ Moon Dev Indicators Initialized! Ready for launch! 🚀")

    def next(self):
        # 🌑 Check if we're in cooldown from losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            print("🌪️ Moon Dev Alert: Trading paused - 3 consecutive losses! ✋")
            return

        # 🚀 Entry Logic
        if not self.position:
            # Calculate squeeze condition
            upper = self.upper_band[-1]
            lower = self.lower_band[-1]
            squeeze_threshold = 0.1  # 10%
            
            if upper > 0 and lower > 0:
                squeeze_ratio = (upper - lower) / ((upper + lower) / 2)
                squeeze = squeeze_ratio <= squeeze_threshold
            else:
                squeeze = False

            # Volume confirmation
            obv_condition = self.obv[-1] >= self.obv_high[-1]
            
            # Price breakout
            price = self.data.Close[-1]
            long_breakout = price > upper
            short_breakout = price < lower

            # Execute entries
            if squeeze and obv_condition:
                if long_breakout:
                    self.enter_long(price)
                elif short_breakout:
                    self.enter_short(price)

        # 🛑 Exit Logic
        else:
            if self.position.is_long:
                self.exit_long()
            else:
                self.exit_short()

    def enter_long(self, price):
        # Calculate position size 🌙
        size = (self.risk_per_trade * self.equity) / price
        size = int(round(size))
        
        if size > 0:
            self.buy(size=size)
            print(f"🚀 LONG LAUNCH! {size} units @ {price:.2f} | Equity: {self.equity:,.2f}")
            # Initialize trailing stop
            self.trailing_high = self.data.High[-1]

    def enter_short(self, price):
        size = (self.risk_per_trade * self.equity) / price
        size = int(round(size))
        
        if size > 0:
            self.sell(size=size)
            print(f"🌑 SHORT IGNITION! {size} units @ {price:.2f} | Equity: {self.equity:,.2f}")
            # Initialize trailing stop
            self.trailing_low = self.data.Low[-1]

    def exit_long(self):
        # Update trailing stop
        self.trailing_high = max