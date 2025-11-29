Here's the fixed and completed backtest code with Moon Dev themed debug prints and proper position sizing implementation:

```python
# 🌙 Moon Dev's VolSync Squeeze Backtest 🌙
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

print("🌙 Initializing Moon Dev's VolSync Squeeze Backtest... ✨")

# =====================
# DATA PREPARATION 🌙📊
# =====================
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean and prepare data ✨
print("🌙 Cleaning cosmic data dust...")
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'datetime': 'Date'
})
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

class VolSyncSqueeze(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # =====================
        # INDICATORS 🌙📈
        # =====================
        print("🌙 Initializing celestial indicators...")
        # Bollinger Bands Components
        self.bb_middle = self.I(talib.SMA, self.data.Close, timeperiod=20, name='BB Middle')
        self.bb_stddev = self.I(talib.STDDEV, self.data.Close, timeperiod=20, name='BB StdDev')
        self.bb_upper = self.I(lambda sma, std: sma + 2*std, self.bb_middle, self.bb_stddev, name='BB Upper')
        self.bb_lower = self.I(lambda sma, std: sma - 2*std, self.bb_middle, self.bb_stddev, name='BB Lower')
        
        # Bollinger Bandwidth Calculation 🌙📏
        self.bb_bandwidth = self.I(lambda u,l,m: (u-l)/m, self.bb_upper, self.bb_lower, self.bb_middle, name='BB Bandwidth')
        self.bb_bandwidth_low = self.I(talib.MIN, self.bb_bandwidth, timeperiod=10, name='BB Band Low')
        
        # OBV and Swing Lows 🌙🌊
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume, name='OBV')
        self.price_low = self.I(talib.MIN, self.data.Low, timeperiod=5, name='Price Swing Low')
        self.obv_low = self.I(talib.MIN, self.obv, timeperiod=5, name='OBV Swing Low')
        
        self.pending_entry = False  # Entry flag 🌙🚩

    def next(self):
        # =====================
        # ENTRY LOGIC 🌙🚀
        # =====================
        if not self.position:
            # Detect bullish divergence 🌙📈
            if len(self.data) > 1:
                price_div = self.price_low[-1] < self.price_low[-2]
                obv_div = self.obv_low[-1] > self.obv_low[-2]
                divergence = price_div and obv_div
                
                # Check volatility contraction 🌙🌀
                bandwidth_contraction = self.bb_bandwidth[-1] <= self.bb_bandwidth_low[-1]
                
                # Price position check 🌙📊
                price_above_middle = self.data.Close[-1] > self.bb_middle[-1]
                
                if divergence and bandwidth_contraction and price_above_middle:
                    print(f"🌙✨ CONFLUENCE DETECTED! {self.data.index[-1]} ✨🌙")
                    self.pending_entry = True
            
            # Trigger entry on next green candle 🌙🕯️
            if self.pending_entry and (self.data.Close[-1] > self.data.Open[-1]):
                # Calculate risk parameters 🌙⚖️
                entry_price = self.data.Close[-1]
                stop_loss = max(self.price_low[-1], self.bb_middle[-1])
                risk_amount = self.equ