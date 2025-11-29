Here's the complete implementation of the VolSqueezeReversal strategy for backtesting.py:

```python
# 🌙 Moon Dev's VolSqueezeReversal Backtest 🌙
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib

# ========== DATA PREPROCESSING ==========
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"

# Load and clean data
data = pd.read_csv(data_path)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename and format columns
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert to datetime index
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

# Resample to 4H timeframe 🌙
data = data.resample('4H').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

# Mock funding rate and liquidation data ✨
np.random.seed(42)
data['Funding_Rate'] = np.random.uniform(-0.1, 0.1, len(data))
data['Liquidation_Cluster'] = data['Low'].rolling(20).min()

# ========== STRATEGY CLASS ==========
class VolSqueezeReversal(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌕
    
    def init(self):
        # 🌙 Heikin-Ashi Calculations
        ha_close = (self.data.Open + self.data.High + self.data.Low + self.data.Close) / 4
        ha_open = np.zeros_like(ha_close)
        ha_open[0] = (self.data.Open[0] + self.data.Close[0]) / 2
        for i in range(1, len(ha_close)):
            ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
            
        ha_high = np.maximum.reduce([self.data.High, ha_open, ha_close])
        ha_low = np.minimum.reduce([self.data.Low, ha_open, ha_close])

        self.ha_open = self.I(lambda: ha_open, name='HA_Open')
        self.ha_close = self.I(lambda: ha_close, name='HA_Close')
        self.ha_high = self.I(lambda: ha_high, name='HA_High')
        self.ha_low = self.I(lambda: ha_low, name='HA_Low')

        # 🌗 Bollinger Bands
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, 
                             timeperiod=20, nbdevup=2, nbdevdn=2, 
                             matype=0, name='BB_Upper', index=0)
        self.bb_lower = self.I(talib.BBANDS, self.data.Close,
                             timeperiod=20, nbdevup=2, nbdevdn=2,
                             matype=0, name='BB_Lower', index=2)
        self.bb_width = self.I(lambda: self.bb_upper - self.bb_lower, name='BB_Width')
        self.bb_width_low = self.I(talib.MIN, self.bb_width, 20, name='BB_Width_Low')

        # 🌀 Funding Rate
        self.funding = self.I(lambda: self.data.Funding_Rate, name='Funding_Rate')

        # 💧 Liquidation Zones
        self.liquidation = self.I(talib.MIN, self.data.Low, 20, name='Liquidation_Cluster')

    def next(self):
        if len(self.data) < 20:  # Warmup period
            return

        # 🌙 Current Conditions
        ha_body = abs(self.ha_close[-1] - self.ha_open[-1])
        ha_range = self.ha_high[-1] - self.ha_low[-1]
        body_ratio = ha_body / ha_range if ha_range > 0 else 0
        
        bb_squeeze = self.bb_width[-1] <= self.bb_width_low[-1]
        funding_flip = self.funding[-1] < 0