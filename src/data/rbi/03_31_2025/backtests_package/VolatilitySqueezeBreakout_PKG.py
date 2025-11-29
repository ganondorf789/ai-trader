I'll fix the code by removing all `backtesting.lib` imports and replacing crossover/crossunder functions with manual implementations. Here's the corrected version with Moon Dev themed improvements:

```python
# 🌙 Moon Dev's Volatility Squeeze Breakout Backtest 🌙
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# Clean and prepare moon data 🌕
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map columns to moon chart standards 🌑
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolatilitySqueezeBreakout(Strategy):
    # Lunar configuration parameters 🌓
    bb_length = 20
    bb_std = 2
    donchian_length = 20
    atr_length = 14
    lookback_period = 35040  # 1 year in 15m bars 🌗
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌌 Cosmic indicator calculations
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume
        
        # Volume-Weighted Bollinger Bands 🌐
        self.vwma = self.I(ta.vwma, close, volume, self.bb_length, name='VWMA')
        self.close_vwma_diff = self.I(lambda c, v: c - v, close, self.vwma, name='Close-VWMA')
        self.std = self.I(talib.STDDEV, self.close_vwma_diff, self.bb_length, name='STD')
        self.upper_band = self.I(lambda v, s: v + s*self.bb_std, self.vwma, self.std, name='UPPER')
        self.lower_band = self.I(lambda v, s: v - s*self.bb_std, self.vwma, self.std, name='LOWER')
        
        # Volatility Squeeze Detection 🌪️
        self.bandwidth = self.I(lambda u, l, v: (u - l)/v, 
                              self.upper_band, self.lower_band, self.vwma, name='BANDWIDTH')
        self.bandwidth_min = self.I(talib.MIN, self.bandwidth, self.lookback_period, name='BAND_MIN')
        
        # Breakout Channels 🌄
        self.donchian_upper = self.I(talib.MAX, high, self.donchian_length, name='DONCHIAN_UPPER')
        self.donchian_lower = self.I(talib.MIN, low, self.donchian_length, name='DONCHIAN_LOWER')
        
        # Exit Mechanism 🌊
        self.atr = self.I(talib.ATR, high, low, close, self.atr_length, name='ATR')
        
        # Moon tracking variables 🌖
        self.max_high = None
        self.min_low = None

    def next(self):
        # 🌠 Lunar debug console
        if len(self.data) % 1000 == 0:
            print(f"🌙 Scanning bar {len(self.data)} | Price: {self.data.Close[-1]:.2f} | Equity: {self.equity:,.2f} ✨")
        
        if not self.position:
            # 🌗 Check for cosmic squeeze condition
            squeeze_active = abs(self.bandwidth[-1] - self.bandwidth_min[-1]) < 1e-5
            
            # 🚀 Long entry: Squeeze + Breakout
            if squeeze_active and (self.data.Close[-2] < self.donchian_upper[-2] and 
                                 self.data.Close[-1] > self.donchian_upper[-1]):
                atr_value = self.atr[-1]
                if atr_value > 0:
                    position_size = int(round((self.risk_percent * self.equity) / (3 * atr_value)))
                    if position_size > 0:
                        self.buy(size