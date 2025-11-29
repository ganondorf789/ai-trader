```python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Data preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean and prepare data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolSqueezeBreakout(Strategy):
    bb_period = 20
    bb_dev = 2
    lookback_percentile = 100
    volume_z_threshold = 2
    risk_percent = 1
    max_holding_bars = 480  # 5 days in 15m intervals

    def init(self):
        # Calculate Bollinger Bands
        close = self.data.Close.values
        self.upper, self.middle, self.lower = talib.BBANDS(
            close, 
            timeperiod=self.bb_period,
            nbdevup=self.bb_dev,
            nbdevdn=self.bb_dev,
            matype=0
        )
        
        # Add indicators using self.I()
        self.bb_upper = self.I(self._wrap(self.upper), name='BB_Upper')
        self.bb_middle = self.I(self._wrap(self.middle), name='BB_Middle')
        self.bb_lower = self.I(self._wrap(self.lower), name='BB_Lower')

        # Calculate BB Width Percentile
        bb_width = (self.upper - self.lower) / self.middle
        self.bbwp = self.I(self._calc_bbwp(bb_width), name='BBWP')

        # Calculate Volume Z-Score
        volume = self.data.Volume.values
        ma_volume = talib.SMA(volume, timeperiod=20)
        std_volume = talib.STDDEV(volume, timeperiod=20)
        self.volume_z = self.I(self._wrap((volume - ma_volume) / std_volume), name='Volume_Z')

    def _wrap(self, array):
        """Wrap array in lambda for self.I()"""
        return lambda: array

    def _calc_bbwp(self, bb_width):
        """Calculate rolling percentile for BB Width"""
        bbwp = np.full_like(bb_width, np.nan)
        for i in range(self.lookback_percentile, len(bb_width)):
            window = bb_width[i-self.lookback_percentile:i]
            bbwp[i] = np.percentile(window, 10)
        return bbwp

    def next(self):
        if not self.position:
            # Entry logic
            current_bbwp = self.bbwp[-1]
            current_volume_z = self.volume_z[-1]
            price_close = self.data.Close[-1]

            if (current_bbwp < 10 and 
                current_volume_z > self.volume_z_threshold):

                # Long entry
                if price_close > self.bb_upper[-1]:
                    swing_low = talib.MIN(self.data.Low, timeperiod=20)[-1]
                    self._enter_trade('long', swing_low)

                # Short entry
                elif price_close < self.bb_lower[-1]:
                    swing_high = talib.MAX(self.data.High, timeperiod=20)[-1]
                    self._enter_trade('short', swing_high)
        else:
            # Exit logic
            current_bbwp = self.bbwp[-1]
            price_close = self.data.Close[-1]

            # Volatility expansion exit
            if current_bbwp > 90:
                self.position.close()
                print(f"🌙✨🌑 Moon Dev Exit: BBWP {current_bbwp:.1f} > 90th percentile!")

            # Price back inside bands
            elif self.bb_lower[-1] < price_close < self.bb_upper[-1]:
                self.position.close()
                print(f"🌙✨🌒 Moon Dev Exit: Price {price_close} inside Bands!")

            # Max holding period
            elif self.position.duration >= self.max_holding_bars:
                self.position.close()
                print("