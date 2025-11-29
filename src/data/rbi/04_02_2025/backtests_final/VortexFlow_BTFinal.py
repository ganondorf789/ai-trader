Here's the fixed and complete backtest code with Moon Dev themed debug prints and proper position sizing:

```python
import pandas as pd
from backtesting import Strategy, Backtest
import talib
import pandas_ta as ta
import numpy as np

# Data Handling
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VortexFlow(Strategy):
    def init(self):
        # Moon Dev Initialization Sequence 🌙
        print("🌙 Initializing Moon Dev Trading System...")
        
        # Clean data columns
        self.data.columns = [col.strip().lower() for col in self.data.columns]
        
        # 1. Vortex Indicator (VI+, VI-)
        vortex = self.data.ta.vortex(length=14)
        self.vi_plus = self.I(lambda: vortex['VIPT_14'], name='VI+')
        self.vi_minus = self.I(lambda: vortex['VIMN_14'], name='VI-')
        print("🌙 Vortex Indicators Activated!")
        
        # 2. Bollinger Bands + Bandwidth
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        bb_bandwidth = (self.bb_upper - self.bb_lower) / self.bb_middle
        self.bb_bandwidth_sma = self.I(talib.SMA, bb_bandwidth, 20)
        print("🌙 Bollinger Bands Online!")
        
        # 3. Funding Rate Momentum
        if 'funding_rate' in self.data.columns:
            self.funding_roc = self.I(talib.ROC, self.data['funding_rate'], 3)
            print("🌙 Funding Rate Momentum Scanner Active!")
        else:
            print("⚠️ Moon Dev Warning: Funding Rate data not found")
        
        # 4. Netflow Divergence Indicators
        if 'netflow' in self.data.columns:
            self.swing_low = self.I(talib.MIN, self.data.Low, 20)
            self.netflow_low = self.I(talib.MIN, self.data['netflow'], 20)
            self.swing_high = self.I(talib.MAX, self.data.High, 20)
            self.netflow_high = self.I(talib.MAX, self.data['netflow'], 20)
            print("🌙 Netflow Divergence Detectors Ready!")
        else:
            print("⚠️ Moon Dev Warning: Netflow data not found")
        
        # Volatility Stop (ATR)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        print("🌙 Volatility Protection Systems Armed!")
        
    def next(self):
        # Moon Dev Debug Prints 🌙
        print(f"\n🌙 Moon Dev Status Update 🌙\nCurrent Close: {self.data.Close[-1]}")
        print(f"🌙 Current Equity: {self.equity:.2f}")
        
        # Entry Conditions
        vi_cross_up = (self.vi_plus[-2] <= self.vi_minus[-2]) and (self.vi_plus[-1] > self.vi_minus[-1])
        vi_cross_down = (self.vi_plus[-2] >= self.vi_minus[-2]) and (self.vi_plus[-1] < self.vi_minus[-1])
        
        # Netflow Divergence Checks
        bull_div = False
        bear_div = False
        
        if len(self.swing_low) > 21 and hasattr(self, 'netflow_low'):
            bull_div = (self.swing_low[-1] < self.swing_low[-21]) and (self.netflow_low[-1] > self.netflow_low[-21])
            
        if len(self.swing_high) > 21 and hasattr(self, 'netflow_high'):
            bear_div = (self.swing_high[-1] > self.swing_