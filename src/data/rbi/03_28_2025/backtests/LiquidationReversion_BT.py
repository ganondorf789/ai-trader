```python
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib
import pandas_ta as ta

class LiquidationReversion(Strategy):
    ema_period = 20
    atr_period = 20
    keltner_multiplier = 2
    swing_period = 20
    risk_percent = 0.01
    volume_ma_period = 20

    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period, name='SWING HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period, name='SWING LOW')
        self.ema = self.I(talib.EMA, self.data.Close, self.ema_period, name='EMA')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR')
        self.keltner_upper = self.I(lambda ema, atr: ema + self.keltner_multiplier * atr, self.ema, self.atr, name='KELTNER UPPER')
        self.keltner_lower = self.I(lambda ema, atr: ema - self.keltner_multiplier * atr, self.ema, self.atr, name='KELTNER LOWER')
        self.volume_ma = self.I(talib.SMA, self.data.Volume, self.volume_ma_period, name='VOLUME MA')
        
        # Precompute cumulative sums for VWAP calculation
        typical_price = (self.data.High + self.data.Low + self.data.Close) / 3
        tpv = typical_price * self.data.Volume
        self.cum_tpv = np.cumsum(tpv)
        self.cum_vol = np.cumsum(self.data.Volume)
        
        self.last_swing_high_idx = -1
        self.last_swing_low_idx = -1

    def next(self):
        current_idx = len(self.data) - 1
        
        # Detect new swing points
        if self.data.High[-1] == self.swing_high[-1]:
            self.last_swing_high_idx = current_idx
            print(f"🌙🚨 NEW SWING HIGH DETECTED AT {self.data.index[-1]}")
        if self.data.Low[-1] == self.swing_low[-1]:
            self.last_swing_low_idx = current_idx
            print(f"🌙🚨 NEW SWING LOW DETECTED AT {self.data.index[-1]}")
        
        # Calculate anchored VWAP
        vwap_long = vwap_short = None
        current_close = self.data.Close[-1]
        current_atr = self.atr[-1]
        
        # Long setup (swing low anchor)
        if self.last_swing_low_idx != -1:
            anchor_idx = self.last_swing_low_idx
            if anchor_idx == 0:
                cum_tpv = self.cum_tpv[current_idx]
                cum_vol = self.cum_vol[current_idx]
            else:
                cum_tpv = self.cum_tpv[current_idx] - self.cum_tpv[anchor_idx-1]
                cum_vol = self.cum_vol[current_idx] - self.cum_vol[anchor_idx-1]
            if cum_vol > 0:
                vwap_long = cum_tpv / cum_vol
                print(f"🌙💧 LONG VWAP: {vwap_long} (Anchor: {self.data.index[anchor_idx]})")
        
        # Short setup (swing high anchor)
        if self.last_swing_high_idx != -1:
            anchor_idx = self.last_swing_high_idx
            if anchor_idx == 0:
                cum_tpv = self.cum_tpv[current_idx]
                cum_vol = self.cum_vol[current_idx]
            else:
                cum_tpv = self.cum_tpv[current_idx] - self.cum_tpv[anchor_idx-1]
                cum_vol = self.cum_vol[current_idx] - self.cum_vol[anchor_idx-1]
            if cum_vol