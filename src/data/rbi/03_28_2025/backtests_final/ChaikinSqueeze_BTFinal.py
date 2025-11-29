Here's the debugged backtest code with Moon Dev themed improvements while maintaining the original strategy logic:

```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

class ChaikinSqueeze(Strategy):
    risk_pct = 0.01  # 1% of equity per trade (proper fraction format)
    
    def init(self):
        # 🌙 Calculate Chaikin Money Flow using TA-Lib (corrected ADOSC parameters)
        self.cmf = self.I(talib.ADOSC, 
                         self.data.High, 
                         self.data.Low, 
                         self.data.Close, 
                         self.data.Volume, 
                         fastperiod=3, 
                         slowperiod=10)
        
        # 🌀 Calculate Bollinger Bands with proper TA-Lib integration
        upper, middle, lower = talib.BBANDS(self.data.Close, 
                                          timeperiod=20, 
                                          nbdevup=2, 
                                          nbdevdn=2, 
                                          matype=0)
        self.upper_band = self.I(lambda: upper)
        self.middle_band = self.I(lambda: middle)
        self.lower_band = self.I(lambda: lower)
        
        # 📏 Calculate Bollinger Bandwidth with proper array handling
        self.bandwidth = self.I(
            lambda: (np.array(upper) - np.array(lower)) / np.array(middle)
        self.sma_bandwidth = self.I(talib.SMA, self.bandwidth, timeperiod=10)
        
        print("🌕✨ Moon Dev Indicators Activated! TA-Lib systems nominal... 🚀")

    def next(self):
        # Wait for sufficient data 🌙 (minimum 30 bars)
        if len(self.data) < 30:
            return

        # Entry Logic 🌙✨ (using proper array indexing)
        if not self.position:
            # CMF crossover check (previous 2 periods)
            cmf_cross = (self.cmf[-3] < 0) and (self.cmf[-2] > 0)
            
            # Squeeze condition check
            squeeze_condition = (self.bandwidth[-2] <= 0.5 * self.sma_bandwidth[-2])
            
            if cmf_cross and squeeze_condition:
                # 🚀 Calculate position size with proper rounding
                entry_price = self.data.Close[-1]  # Using close instead of open for consistency
                position_size = round((self.risk_pct * self.equity) / entry_price)
                position_size = max(1, int(position_size))  # Ensure at least 1 unit
                
                if position_size > 0:
                    self.buy(size=position_size)
                    print(f"🌙✨ BUY SIGNAL! Size: {position_size} units @ {entry_price:.2f} | Equity: {self.equity:,.2f} 🌌")

        # Exit Logic 🌘 (using proper price levels)
        if self.position:
            # Take profit at upper band (price level)
            if self.data.High[-1] >= self.upper_band[-1]:
                self.position.close()
                print(f"🎯 PROFIT TAKEN @ {self.upper_band[-1]:.2f} | PNL: {self.position.pl:.2f} 🌕")
            
            # Stop loss on CMF cross below zero (price level)
            if (self.cmf[-2] > 0) and (self.cmf[-1] < 0):
                self.position.close()
                print(f"🛑 CMF STOP TRIGGERED! Exiting @ {self.data.Close[-1]:.2f} | PNL: {self.position.pl:.2f} 🌘")

# Data preparation 🌙 (fixed column formatting)
data = pd.read_csv("/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv")
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col], errors='ignore')

# Format columns for backtesting.py 🌙 (consistent capitalization)
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data.index = pd.to_datetime(data['datetime'])
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]  # Ensure proper column order

# Launch backtest 🚀 (fixed commission format)
bt = Backtest(data, ChaikinSqueeze, cash=1_000_000, commission=.