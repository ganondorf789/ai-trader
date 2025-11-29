I'll analyze and fix the code while maintaining the original strategy logic. Here's the corrected version with Moon Dev themed improvements:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib
from talib import MA_Type

# Data preprocessing - Moon Dev style data cleansing 🌙
print("🌙 Initiating lunar data alignment sequence...")
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and prepare data with cosmic precision
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col], errors='ignore')
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

print("🌌 Data successfully harmonized with lunar cycles!")

class ClusterVoltract(Strategy):
    # Strategy parameters - calibrated with moon gravity precision
    bb_period = 20
    bb_dev = 2
    swing_period = 20
    volatility_avg_period = 20
    funding_extreme = 5  # 5th/95th percentiles
    entry_distance = 2  # Y% from swing low
    vol_expansion = 20  # Z% expansion threshold
    max_risk = 1  # C% of equity
    stop_buffer = 1  # A% below swing low
    funding_retrace = 50  # B% retracement
    max_hold_bars = 20  # D bars

    def init(self):
        # Calculate core indicators with celestial precision
        close = self.data.Close
        low = self.data.Low
        
        # Bollinger Bands components
        self.upper, self.middle, self.lower = self.I(talib.BBANDS, close, 
            timeperiod=self.bb_period, nbdevup=self.bb_dev, nbdevdn=self.bb_dev, 
            matype=MA_Type.T3, name=['BB_Upper','BB_Mid','BB_Lower'])
        
        # Volatility calculations
        bb_width = (self.upper - self.lower) / self.middle
        self.volatility = self.I(bb_width, name='BB_Width')
        self.volatility_avg = self.I(talib.SMA, bb_width, self.volatility_avg_period, 
            name='Volatility_Avg')

        # Swing low detection
        self.swing_lows = self.I(talib.MIN, low, self.swing_period, name='Swing_Low')

        # Funding rate analysis
        funding = self.data.funding_rate
        self.fund_low = funding.expanding().apply(lambda x: np.percentile(x, self.funding_extreme))
        self.fund_high = funding.expanding().apply(lambda x: np.percentile(x, 100-self.funding_extreme))

    def next(self):
        current_idx = len(self.data) - 1
        
        # Entry conditions
        if not self.position:
            # Moon-themed debug prints 🌙
            if self.data.funding_rate[current_idx] <= self.fund_low[current_idx]:
                print(f"🌙 Funding Rate DIP detected! Lunar cycle favorable ({self.data.funding_rate[current_idx]:.4f} ≤ {self.fund_low[current_idx]:.4f})")
            elif self.data.funding_rate[current_idx] >= self.fund_high[current_idx]:
                print(f"🌙 Funding Rate SPIKE detected! Moon gravity pulling prices ({self.data.funding_rate[current_idx]:.4f} ≥ {self.fund_high[current_idx]:.4f})")
            
            if (hasattr(self.data, 'liquidation_cluster') and self.data.liquidation_cluster[current_idx] and \
               self.volatility[current_idx] < self.volatility_avg[current_idx] and \
               ((self.data.Close[current_idx] - self.swing_lows[current_idx])/self.swing_lows[current_idx]) <= (self.entry_distance/100):
                
                # Risk management calculations with cosmic precision
                stop_price = self.swing_lows[current_idx] * (1 - self.stop_buffer/100)
                risk_per_unit = abs(self.data.Close[current_idx] - stop_price)
                position_size = round((self.equity * (self.max_risk/100))