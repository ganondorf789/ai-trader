To address the problem, we'll create a backtesting implementation for the BandwidthBreakout strategy following the specified requirements. Here's the structured solution:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import talib
import os

# Data preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class BandwidthBreakout(Strategy):
    def init(self):
        # Initialize indicators
        self.upper_band, self.middle_band, self.lower_band = self.I(
            talib.BBANDS, self.data.Close, 
            timeperiod=20, nbdevup=2, nbdevdn=2, matype=0,
            name=['UpperBB', 'MiddleBB', 'LowerBB']
        )
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20, name='Volume_MA')
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, 14, name='ADX')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        
        # Track bandwidth and trailing levels
        self.bandwidth_history = []
        self.entry_bandwidth = None
        self.trailing_high = None
        self.trailing_low = None

    def next(self):
        # Skip initial bars without full indicator data
        if len(self.data.Close) < 20:
            return

        # Calculate current bandwidth
        current_upper = self.upper_band[-1]
        current_lower = self.lower_band[-1]
        current_middle = self.middle_band[-1]
        current_bandwidth = (current_upper - current_lower) / current_middle
        self.bandwidth_history.append(current_bandwidth)

        # Calculate bandwidth's moving average
        bb_width_ma = np.nan
        if len(self.bandwidth_history) >= 20:
            bb_width_ma = talib.SMA(np.array(self.bandwidth_history), 20)[-1]

        # Get current market conditions
        current_volume = self.data.Volume[-1]
        current_volume_ma = self.volume_ma[-1]
        current_adx = self.adx[-1]
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]

        # Entry logic
        if not self.position:
            valid_ma = not np.isnan(bb_width_ma)
            bb_contraction = current_bandwidth < bb_width_ma if valid_ma else False
            volume_spike = current_volume > 1.5 * current_volume_ma
            adx_strong = current_adx > 25

            if bb_contraction and volume_spike and adx_strong:
                # Long entry
                if current_close > current_upper:
                    risk_percent = 0.01
                    sl_price = current_lower
                    risk_amount = self.equity * risk_percent
                    risk_per_share = current_close - sl_price
                    
                    if risk_per_share > 0:
                        position_size = int(round(risk_amount / risk_per_share))
                        self.entry_bandwidth = current_bandwidth
                        self.trailing_high = current_high
                        atr_sl = self.trailing_high - 2 * self.atr[-1]
                        self.buy(size=position_size, sl=atr_sl)
                        print(f"🌙🚀 LONG Entry: {position_size} units @ {current_close}, SL: {atr_sl:.2f}")

                # Short entry        
                elif current_close < current_lower:
                    risk_percent = 0.01
                    sl_price = current_upper
                    risk_amount = self.equity * risk_percent
                    risk_per_