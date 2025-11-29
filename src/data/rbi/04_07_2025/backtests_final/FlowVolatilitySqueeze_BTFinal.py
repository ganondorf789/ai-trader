I'll debug the code while maintaining the original strategy logic. Here's the fixed version with Moon Dev-themed improvements:

```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

# Data preparation and cleaning
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Validate required columns
assert 'netflow' in data.columns, "🌑 Moon Dev Alert! Netflow data missing - cosmic disturbance detected"

class FlowVolatilitySqueeze(Strategy):
    risk_pct = 0.01  # 1% capital risk per trade 🚀
    bb_period = 20
    atr_period = 14
    max_hold_bars = 288  # 72 hours in 15m timeframe
    
    def init(self):
        # 🌙 Volatility Indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        # 🌌 Bollinger Bands Setup
        self.bb_upper = self.I(lambda c: talib.BBANDS(c, self.bb_period, 2, 2)[0], self.data.Close)
        self.bb_lower = self.I(lambda c: talib.BBANDS(c, self.bb_period, 2, 2)[2], self.data.Close)
        self.bb_width = self.I(lambda u,l: u-l, self.bb_upper, self.bb_lower)
        self.bb_std = self.I(talib.STDDEV, self.bb_width, self.bb_period)
        
        # 🌊 Netflow Analysis
        self.nf_mean = self.I(talib.SMA, self.data.netflow, 20)  # Fixed lowercase netflow
        self.nf_std = self.I(talib.STDDEV, self.data.netflow, 20)  # Fixed lowercase netflow
        self.nf_z = self.I(lambda n,s,d: (n-s)/d, self.data.netflow, self.nf_mean, self.nf_std)  # Fixed lowercase netflow
        
        self.entry_bar = 0  # 🌑 Track entry timing

    def next(self):
        price = self.data.Close[-1]
        
        # 🌙 Volatility Filter
        if self.atr[-1] < 0.02 * price:
            print("🌑 Moon Dev Alert: Market too calm - skipping trade")
            return  # Skip trades in calm markets
            
        if not self.position:
            # 🚀 Entry Conditions
            bb_trigger = self.bb_width[-1] > 2 * self.bb_std[-1]
            flow_trigger = self.nf_z[-1] < -2
            
            if bb_trigger and flow_trigger:
                # 🌑 Calculate position size
                risk_amount = self.equity * self.risk_pct
                stop_distance = 1.5 * self.bb_width[-1]
                position_size = int(round(risk_amount / stop_distance))
                
                if position_size > 0:
                    self.sell(size=position_size, 
                             sl=price + stop_distance,
                             tag="MoonDev_Short")
                    self.entry_bar = len(self.data)
                    print(f"🚀🌙 MOON DEV SHORT ACTIVATED! Size: {position_size} @ {price:.2f}")
        
        else:
            # ✨ Exit Conditions
            hold_time = len(self.data) - self.entry_bar
            vol_exit = self.bb_width[-1] < 1 * self.bb_std[-1]
            flow_exit = self.nf_z[-1] > 0
            
            if vol_exit or flow_exit or hold_time >= self.max_hold_bars:
                self.position.close()
                reason = []
                if vol_exit: reason.append("Vol Squeeze 🌪")
                if flow_exit: reason.append("Flow Reversal 🌊")
                if hold_time >= self.max_hold_bars: reason.append("Time Expired ⏳")
                print(f"🌑✨ MOON DEV EXIT: {',