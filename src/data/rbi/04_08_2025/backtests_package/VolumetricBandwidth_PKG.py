Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed debug prints added:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from datetime import time

# Moon Dev's VolumetricBandwidth Strategy 🌙
class VolumetricBandwidth(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    atr_period = 14
    bb_period = 20
    bb_std = 2
    volume_avg_period = 20
    bandwidth_median_period = 14
    liquidity_hours = (time(13, 0), time(17, 0))  # London/NY overlap in UTC

    def init(self):
        # ======== INDICATORS ======== 🌙
        # Volume Analysis
        self.volume_avg = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_avg_period, name='Volume_MA')
        
        # Bollinger Bands
        def bb_upper(close):
            upper, _, _ = talib.BBANDS(close, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std)
            return upper
        self.bb_upper = self.I(bb_upper, self.data.Close, name='BB_Upper')
        
        def bb_middle(close):
            _, middle, _ = talib.BBANDS(close, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std)
            return middle
        self.bb_middle = self.I(bb_middle, self.data.Close, name='BB_Middle')
        
        # Bandwidth Calculation
        def bandwidth(upper, lower, middle):
            return (upper - lower) / middle
        bb_lower = self.I(lambda close: talib.BBANDS(close, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std)[2], self.data.Close, name='BB_Lower')
        self.bandwidth = self.I(bandwidth, self.bb_upper, bb_lower, self.bb_middle, name='Bandwidth')
        
        # Bandwidth Median
        self.bandwidth_median = self.I(lambda x: ta.median(x, length=self.bandwidth_median_period), self.bandwidth, name='Bandwidth_Median')
        
        # Volatility Measure
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period, name='ATR')
        
        # ======== TRADE TRACKERS ======== 🚀
        self.last_day = None
        self.daily_max_loss = None

    def next(self):
        # ======== RISK MANAGEMENT ======== 🛡️
        # Daily loss check
        current_date = self.data.index[-1].date()
        if current_date != self.last_day:
            self.last_day = current_date
            self.daily_max_loss = self.equity * 0.95
            print(f"🌙 New Moon Day {current_date} | Equity: ${self.equity:,.2f} | Max Loss: ${self.daily_max_loss:,.2f}")
        
        if self.equity < self.daily_max_loss:
            print(f"🚨 COSMIC COLLAPSE! Trading halted at {self.data.index[-1]} 🌑")
            return
            
        # Liquidity hours filter
        current_time = self.data.index[-1].time()
        if not (self.liquidity_hours[0] <= current_time < self.liquidity_hours[1]):
            return

        # ======== SIGNAL CALCULATION ======== 📡
        current_volume = self.data.Volume[-1]
        volume_signal = current_volume > 2 * self.volume_avg[-1]
        bandwidth_signal = self.bandwidth[-1] > 1.5 * self.bandwidth_median[-1]
        price_position = self.data.Close[-1] > self.bb_middle[-1]

        # ======== EXECUTION LOGIC ======== ⚡
        # Exit on volume contraction
        if self.position and (current_volume < self.volume_avg[-1]):
            print(f"🌙 VOLUME BLACKOUT! Closing position at {self.data.index[-1]}")
            self.position.close()
            return

        # Long Entry
        if not self.position