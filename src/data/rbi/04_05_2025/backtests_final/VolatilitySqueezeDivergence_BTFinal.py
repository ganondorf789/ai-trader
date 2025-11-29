Here's the fully debugged and completed backtest code with Moon Dev cosmic enhancements ✨🌙:

```python
# -*- coding: utf-8 -*-
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Load and preprocess data with Moon Dev cosmic energy ✨
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean column names with lunar precision 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

# Rename columns to match backtesting requirements
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Verify required columns with cosmic checks 🌌
assert 'fundingrate' in data.columns, "🌙 CRITICAL: Funding rate data missing! Add 'funding_rate' column ✨"
assert all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']), "🌙 Missing OHLCV columns"

class VolatilitySqueezeDivergence(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    lookback = 5  # Divergence lookback period
    
    def init(self):
        # Calculate Bollinger Bands with celestial precision 🌕
        self.bb_upper = self.I(self._bb_upper, self.data.Close)
        self.bb_lower = self.I(self._bb_lower, self.data.Close)
        
        # Calculate bandwidth metrics with lunar cycles 🌓
        self.bandwidth = self.I(lambda: self.bb_upper - self.bb_lower, name='Bandwidth')
        self.bandwidth_sma = self.I(talib.SMA, self.bandwidth, 50, name='Bandwidth_SMA')
        
        # Swing highs/lows for stop placement with gravitational pull 🌍
        self.swing_high = self.I(talib.MAX, self.data.High, 20, name='Swing_High')
        self.swing_low = self.I(talib.MIN, self.data.Low, 20, name='Swing_Low')

    def _bb_upper(self, close):
        upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        return upper
    
    def _bb_lower(self, close):
        _, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        return lower

    def next(self):
        # Wait for sufficient data with cosmic patience 🌌
        if len(self.data) < 50:
            print("🌙 Waiting for cosmic alignment (insufficient data)")
            return
            
        # Get current values with lunar precision 🌗
        price_close = self.data.Close[-1]
        funding = self.data.fundingrate[-1]
        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]
        bandwidth = self.bandwidth[-1]
        bandwidth_sma = self.bandwidth_sma[-1]

        # 1. Volatility contraction check 🌙
        vol_contraction = bandwidth < bandwidth_sma
        if not vol_contraction:
            print("🌙 Volatility too expansive, waiting for contraction")
            return
            
        # 2. Funding rate extremes 🌊
        funding_extreme = abs(funding) > 0.001
        if not funding_extreme:
            print("🌙 Funding rates too neutral, waiting for extremes")
            return
            
        # 3. Divergence detection with cosmic alignment 🔄
        price_lows = self.data.Low[-self.lookback:]
        fund_rates = self.data.fundingrate[-self.lookback:]
        
        lower_lows = all(price_lows[i] > price_lows[i+1] for i in range(self.lookback-1))
        rising_funding = all(fund_rates[i] < fund_rates[i+1] for i in range(self.lookback-1))
        bearish_div = lower_lows and rising_funding and funding > 0
        
        higher_highs = all(price_lows[i] < price_lows[i+1] for i in range(self.lookback-1))
        falling_funding = all(fund_rates