I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete, debugged version with Moon Dev themed prints:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np
from backtesting import Strategy
from backtesting import Backtest

# Load and prepare data
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data[['open', 'high', 'low', 'close', 'volume']]
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

class BandSqueezeBreakout(Strategy):
    risk_percent = 0.01
    max_positions = 5
    
    def init(self):
        # Bollinger Bands
        self.upper_bb = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name='Upper BB', which=0)
        self.lower_bb = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name='Lower BB', which=2)
        
        # Keltner Channel
        self.ema_20 = self.I(talib.EMA, self.data.Close, timeperiod=20)
        self.atr_20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=20)
        self.upper_kc = self.I(lambda e,a: e + 1.5*a, self.ema_20, self.atr_20)
        self.lower_kc = self.I(lambda e,a: e - 1.5*a, self.ema_20, self.atr_20)
        
        # Volume SMA
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
        # CVI Indicator
        def compute_cvi(h, l, length):
            return ta.cvi(h, l, length=length)
        self.cvi = self.I(compute_cvi, self.data.High, self.data.Low, 9)
        
        # RSI
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14)
        
        # Squeeze Condition
        self.squeeze = self.I(lambda ub,uk,lb,lk: (ub < uk) & (lb > lk),
                            self.upper_bb, self.upper_kc, self.lower_bb, self.lower_kc)
        
        print("🌙 MOON DEV INIT: Strategy engines primed for launch! 🚀✨")
        print("🌙 INDICATOR STATUS:")
        print(f"   - Bollinger Bands: ONLINE (20,2,2)")
        print(f"   - Keltner Channel: ONLINE (EMA20, ATR20)")
        print(f"   - Volume SMA: ONLINE (20)")
        print(f"   - CVI: ONLINE (9)")
        print(f"   - RSI: ONLINE (14)")
        print("🌙 All systems nominal! Ready for lunar trading! 🌕")

    def next(self):
        # Skip early bars
        if len(self.data) < 24 or len(self.positions) >= self.max_positions:
            return
        
        # Check for squeeze condition
        squeeze_on = self.squeeze[-1]
        
        # Check volume spike (2x average)
        volume_spike = self.data.Volume[-1] > 2 * self.vol_sma[-1]
        
        # Check breakout conditions
        price_above_upper_bb = self.data.Close[-1] > self.upper_bb[-1]
        price_below_lower_bb = self.data.Close[-1] < self.lower_bb[-1]
        
        # Set pending entry if conditions met
        if squeeze_on and volume_spike:
            if price_above_upper_bb and self.rsi[-1] > 50:
                sl = self.data.Low[-1]
                self.pending_entry = {
                    'direction': 'LONG',
                    'sl': sl
                }
                print("🌙 MOON SIGNAL: Bullish breakout detected! Awaiting entry...")
                
            elif price_below_lower_