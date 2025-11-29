Here's the fully debugged and Moon Dev optimized version of your VoltaSqueeze strategy with all technical issues fixed while preserving the original strategy logic:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

class VoltaSqueeze(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # Clean data and prepare columns - Cosmic Data Purifier 🌌✨
        if hasattr(self.data, 'df'):
            self.data.df.columns = self.data.df.columns.str.strip().str.lower()
            self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns 
                                                    if 'unnamed' in str(col).lower()])
        
        # Calculate indicators using TA-Lib with self.I() wrapper
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume
        
        # Bollinger Bands (20,2) - Cosmic Expansion Indicator 🌌
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        self.upper_band = self.I(lambda x: upper, name='UPPER')
        self.lower_band = self.I(lambda x: lower, name='LOWER')
        
        # Bollinger Band Width Percentile - Moon Phase Compression 🌑➡️🌕
        bb_width = (upper - lower) / middle
        self.bbwp = self.I(lambda x: ta.percentrank(pd.Series(bb_width), length=100), name='BBWP')
        
        # On-Balance Volume - Lunar Tidal Indicator 🌊
        self.obv = self.I(lambda x: talib.OBV(close, volume), name='OBV')
        
        # Swing detection (5-period lookback) - Celestial Swing Detector 🌠
        self.high_swings = self.I(lambda x: talib.MAX(high, timeperiod=5), name='SWING_HIGH')
        self.low_swings = self.I(lambda x: talib.MIN(low, timeperiod=5), name='SWING_LOW')
        self.obv_high = self.I(lambda x: talib.MAX(self.obv, timeperiod=5), name='OBV_HIGH')
        self.obv_low = self.I(lambda x: talib.MIN(self.obv, timeperiod=5), name='OBV_LOW')
        
    def next(self):
        # Wait for sufficient data - Moon Phase Alignment Check 🌓
        if len(self.data.Close) < 100 or len(self.bbwp) < 1:
            return
        
        current_close = self.data.Close[-1]
        bbwp = self.bbwp[-1] if len(self.bbwp) > 0 else 100
        obv = self.obv[-1] if len(self.obv) > 0 else 0
        
        # Moon Dev Divergence Detection 🌛✨
        price_bull_div = (len(self.low_swings) > 2 and 
                         self.low_swings[-1] < self.low_swings[-2] and 
                         len(self.obv_low) > 2 and
                         self.obv_low[-1] > self.obv_low[-2])
        
        price_bear_div = (len(self.high_swings) > 2 and
                         self.high_swings[-1] > self.high_swings[-2] and 
                         len(self.obv_high) > 2 and
                         self.obv_high[-1] < self.obv_high[-2])
        
        # Entry Logic with Moon Dev Flair 🌙🚀
        if not self.position:
            # Long Entry - Moon Rocket Launch Sequence 🚀
            if (bbwp < 20 and 
                len(self.upper_band) > 0 and
                current_close > self.upper_band[-1] and 
                price_bull_div):
                
                sl = self.lower_band[-1] if len(self.lower_band) > 0 else current_close * 0.95
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = abs(current_close - sl)
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=sl)
                    print(f