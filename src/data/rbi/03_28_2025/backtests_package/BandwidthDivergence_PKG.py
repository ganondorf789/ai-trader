Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed indicators:

```python
# -*- coding: utf-8 -*-
from backtesting import Backtest, Strategy
import talib
import pandas as pd
import numpy as np

class BandwidthDivergence(Strategy):
    risk_per_trade = 0.01
    volume_ma_period = 20
    swing_period = 5
    bw_lookback = 100
    bw_threshold = 0.2

    def init(self):
        # Bollinger Bands calculations 🌙
        def bb_upper(close):
            u, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return u
        def bb_middle(close):
            _, m, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return m
        def bb_lower(close):
            _, _, l = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return l
            
        self.bb_upper = self.I(bb_upper, self.data.Close, name='UPPER')
        self.bb_middle = self.I(bb_middle, self.data.Close, name='MIDDLE')
        self.bb_lower = self.I(bb_lower, self.data.Close, name='LOWER')

        # Chaikin Oscillator ✨
        self.chaikin = self.I(talib.ADOSC, self.data.High, self.data.Low, 
                             self.data.Close, self.data.Volume, fastperiod=3, slowperiod=10, name='CHAIKIN')

        # Bandwidth calculations 📈
        self.bandwidth = self.I(lambda: (self.bb_upper - self.bb_lower)/self.bb_middle, 
                              name='BANDWIDTH')
        self.bw_min = self.I(talib.MIN, self.bandwidth, timeperiod=self.bw_lookback, name='BW_MIN')
        self.bw_max = self.I(talib.MAX, self.bandwidth, timeperiod=self.bw_lookback, name='BW_MAX')

        # Volume and swing calculations 🌊
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_ma_period, name='VOL_MA')
        self.price_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period, name='SWING_LOW')
        self.ck_low = self.I(talib.MIN, self.chaikin, timeperiod=self.swing_period, name='CK_LOW')
        self.price_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period, name='SWING_HIGH')
        self.ck_high = self.I(talib.MAX, self.chaikin, timeperiod=self.swing_period, name='CK_HIGH')

        print("🌙 Moon Dev System Activated - Prepare for Lunar Launch! 🚀")

    def next(self):
        if len(self.data) < max(self.bw_lookback, self.swing_period*2):
            return

        # Bandwidth contraction check 📉
        bw_pos = ((self.bandwidth[-1] - self.bw_min[-1]) / 
                 (self.bw_max[-1] - self.bw_min[-1] + 1e-10))
        contraction = bw_pos <= self.bw_threshold

        # Divergence detection 🔄
        bull_div = (self.price_low[-1] > self.price_low[-2] and 
                   self.ck_low[-1] < self.ck_low[-2])
        bear_div = (self.price_high[-1] < self.price_high[-2] and 
                   self.ck_high[-1] > self.ck_high[-2])

        # Volume confirmation 🔊
        vol_ok = self.data.Volume[-1] > self.vol_ma[-1]

        # Trade execution logic 💼
        price = self.data.Close[-1]
        if not self.position:
            if contraction and bull_div and price > self.bb_upper[-1] and vol_ok:
                sl = min(self.bb_lower[-1], self.price_low[-1])
                risk = price - sl
                if risk > 0:
                    size = int(round((self.equity * self.risk_per_trade) / risk))
                    self.buy(size=size, sl=sl, tag='LONG')
                    print(f