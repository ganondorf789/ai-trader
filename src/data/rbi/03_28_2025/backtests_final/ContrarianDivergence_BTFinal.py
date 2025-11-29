I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete, debugged version with Moon Dev improvements:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import numpy as np

class ContrarianDivergence(Strategy):
    def init(self):
        # Moon Dev initialization message
        print("🌙 Initializing Moon Dev's Contrarian Divergence Strategy...")
        print("✨ Preparing lunar indicators for cosmic market analysis...")
        
        # Clean and prepare data
        df = self.data.df
        df.columns = df.columns.str.strip().str.lower()
        df = df.drop(columns=[col for col in df.columns if 'unnamed' in col])
        
        # Calculate indicators with TA-Lib using self.I()
        # Funding Rate Indicators
        funding_rate = df['funding_rate'].values
        lookback_30d = 30 * 24 * 4  # 30 days in 15m intervals
        self.funding_sma_30 = self.I(talib.SMA, funding_rate, timeperiod=lookback_30d, name='Funding SMA 30D')
        self.funding_std_30 = self.I(talib.STDDEV, funding_rate, timeperiod=lookback_30d, nbdev=1, name='Funding STD 30D')
        
        # Exit SMA
        lookback_7d = 7 * 24 * 4
        self.funding_sma_7 = self.I(talib.SMA, funding_rate, timeperiod=lookback_7d, name='Funding SMA 7D')
        
        # RSI
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14, name='RSI 14')
        
        # Swing Highs/Lows
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=5, name='Swing High')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=5, name='Swing Low')
        
        # Track swing extremes
        self.last_swing_high_price = None
        self.last_swing_high_rsi = None
        self.previous_swing_high_price = None
        self.previous_swing_high_rsi = None
        
        self.last_swing_low_price = None
        self.last_swing_low_rsi = None
        self.previous_swing_low_price = None
        self.previous_swing_low_rsi = None

    def next(self):
        # Moon Dev progress tracking
        if len(self.data) % 100 == 0:  # Don't print every tick
            print(f"\n🌙 Processing {self.data.datetime[-1]} - Lunar cycle in progress...")
            print("🔭 Scanning for celestial trading opportunities...")
        
        # Skip initial warmup period
        if len(self.data) < 30*24*4:
            return
        
        # Current values
        current_close = self.data.Close[-1]
        current_funding = self.data.df['funding_rate'][-1]
        sma30 = self.funding_sma_30[-1]
        std30 = self.funding_std_30[-1]
        upper_band = sma30 + 2*std30
        lower_band = sma30 - 2*std30
        
        # Detect swing extremes
        if self.data.High[-1] == self.swing_high[-1]:
            self._update_swing_high(current_close)
        if self.data.Low[-1] == self.swing_low[-1]:
            self._update_swing_low(current_close)
            
        # Check divergence conditions
        bearish_div = self._check_bearish_divergence()
        bullish_div = self._check_bullish_divergence()
        
        # Entry Logic
        if not self.position:
            # Short Entry
            if current_funding > upper_band and bearish_div:
                print("🌘 Bearish divergence detected! Preparing short position...")
                self._enter_short()
            
            # Long Entry
            elif current_funding < lower_band and bullish_div:
                print("🌒 Bullish divergence detected! Preparing long position...")
                self._enter_long()
        
        # Exit Logic
        else:
            self._check_exit_conditions()
    
    def _update_swing_high(self, price):
        # Update swing high tracking
        self.previous_swing_high_price = self.last_swing_high_price
        self.previous_swing_high_rsi = self.last_swing_high_rsi