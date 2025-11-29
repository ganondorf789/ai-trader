I'll fix the code by removing all `backtesting.lib` imports and replacing any crossover functionality with the proper manual implementation. Here's the corrected version with Moon Dev themed debug prints:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Data handling and cleanup
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolumetricSqueeze(Strategy):
    bb_period = 20
    bb_dev = 2
    atr_period = 14
    risk_percent = 0.01
    trailing_stop_multiplier = 1.5
    bb_squeeze_period = 20
    exit_bb_width_multiplier = 2.0

    def init(self):
        # Precompute indicators using TA-Lib
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, timeperiod=self.bb_period, nbdevup=self.bb_dev, nbdevdn=self.bb_dev)
        self.upper_band = self.I(lambda: upper, name='Upper Band')
        self.middle_band = self.I(lambda: middle, name='Middle Band')
        self.lower_band = self.I(lambda: lower, name='Lower Band')
        
        # Bollinger Band Width
        bb_width = (upper - lower) / middle
        self.bb_width = self.I(lambda: bb_width, name='BB Width')
        
        # Min Bollinger Band Width
        min_bb_width = talib.MIN(bb_width, timeperiod=self.bb_squeeze_period)
        self.min_bb_width = self.I(lambda: min_bb_width, name='Min BB Width')
        
        # On-Balance Volume
        obv = talib.OBV(close, volume)
        self.obv = self.I(lambda: obv, name='OBV')
        
        # Average True Range
        atr = talib.ATR(high, low, close, timeperiod=self.atr_period)
        self.atr = self.I(lambda: atr, name='ATR')
        
        self.highest_high = None
        self.entry_bb_width = None

    def next(self):
        # Moon Dev indicator debug
        if len(self.data) % 100 == 0:
            print(f"🌙 Moon Dev Indicators 🌙 | BB Width: {self.bb_width[-1]:.4f} | OBV: {self.obv[-1]:.0f} | ATR: {self.atr[-1]:.2f}")
        
        if not self.position:
            # Entry logic
            if len(self.data) > max(self.bb_period, self.bb_squeeze_period, self.atr_period):
                # Check BB contraction
                bb_contraction = self.bb_width[-1] <= self.min_bb_width[-1]
                
                # Check OBV divergence
                price_divergence = self.data.Low[-1] < self.data.Low[-2]
                obv_divergence = self.obv[-1] > self.obv[-2]
                
                if bb_contraction and price_divergence and obv_divergence:
                    # Risk management calculations
                    atr_value = self.atr[-1]
                    risk_per_share = self.trailing_stop_multiplier * atr_value
                    risk_amount = self.risk_percent * self.equity
                    position_size = risk_amount / risk_per_share
                    position_size = int(round(position_size))
                    
                    if position_size > 0:
                        self.buy(size=position_size)
                        self.highest_high = self.data.High[-1]
                        self.entry_bb_width = self.bb_width[-1]
                        print(f"🚀 MOON DEV ENTRY SIGNAL 🚀 | Size: {position_size} | Price: {self.data.Close[-1]:.2f} | ATR: {atr_value:.2f