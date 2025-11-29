Here's the fully debugged and completed SqueezeMomentum strategy with Moon Dev enhancements:

```python
# 🌙 Moon Dev Backtest AI Certified Code 🚀
# 🚀 Squeeze Momentum Strategy v1.1 - Moon Dev Optimized 🌙

# 1. Necessary imports - Strictly TA-Lib compliant ✨
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# 2. Data handling - Moon Dev Data Preprocessing System ✨
def prepare_data(filepath):
    # Load and clean data with lunar precision
    print("🌙 Moon Dev Data Loading System Initialized...")
    data = pd.read_csv(filepath)
    data.columns = data.columns.str.strip().str.lower()
    
    # Remove any suspicious columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    
    # Standardize column names with Moon Dev formatting
    data.rename(columns={
        'datetime': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    
    # Convert and set index with lunar precision
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    print("✨ Moon Dev Data Preparation Complete! Ready for launch sequence! 🚀")
    return data

# 3. Strategy class - Moon Dev Trading Algorithm 🌙
class SqueezeMomentum(Strategy):
    max_consecutive_losses = 3
    risk_per_trade = 0.02
    trailing_stop_pct = 0.05
    
    def init(self):
        # 🌙 Calculate indicators using TA-Lib only - NO backtesting.lib! 🚫
        # Bollinger Bands with Moon Dev precision
        self.upper_band = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0], self.data.Close)
        self.lower_band = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2], self.data.Close)
        
        # OBV indicators with lunar momentum tracking
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume)
        self.obv_ma = self.I(talib.SMA, self.obv, timeperiod=5)
        self.obv_high = self.I(lambda x: talib.MAX(x, timeperiod=10), self.obv)
        
        # Track trading state with Moon Dev risk management
        self.consecutive_losses = 0
        print("✨ Moon Dev Indicators Initialized! Trading systems nominal! 🚀")

    def next(self):
        # 🌑 Check if we're in cooldown from losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            print("🌪️ Moon Dev Alert: Trading paused - 3 consecutive losses detected! ✋")
            return

        # 🚀 Entry Logic - Moon Dev Signal Detection System
        if not self.position:
            # Calculate squeeze condition with lunar precision
            upper = self.upper_band[-1]
            lower = self.lower_band[-1]
            squeeze_threshold = 0.1  # 10% band width
            
            if upper > 0 and lower > 0:
                squeeze_ratio = (upper - lower) / ((upper + lower) / 2)
                squeeze = squeeze_ratio <= squeeze_threshold
            else:
                squeeze = False

            # Volume confirmation with OBV momentum
            obv_condition = self.obv[-1] >= self.obv_high[-1]
            
            # Price breakout detection
            price = self.data.Close[-1]
            long_breakout = price > upper
            short_breakout = price < lower

            # Execute entries with Moon Dev precision
            if squeeze and obv_condition:
                if long_breakout:
                    self.enter_long(price)
                elif short_breakout:
                    self.enter_short(price)

        # 🛑 Exit Logic - Moon Dev Position Management
        else:
            current_price = self.data.Close[-1]
            
            # Check trailing stop condition
            if self.position.is_long:
                trail_price = self.position.entry_price * (1 - self.trailing_stop_pct)
                if current_price <= trail_price:
                    self.position.close()
                    print("🌙 Moon Dev Exit: Long position