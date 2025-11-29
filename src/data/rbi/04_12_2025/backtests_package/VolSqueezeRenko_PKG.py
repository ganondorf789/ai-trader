I'll fix the code by removing all `backtesting.lib` imports and replacing any usage of its functions with native Python implementations. Here's the corrected version with Moon Dev themed improvements:

```python
import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Moon Dev Data Preparation 🌙✨
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class VolSqueezeRenko(Strategy):
    bb_period = 20
    bb_dev = 2
    atr_period = 14
    lookback_percentile = 100
    volume_threshold = 500
    risk_pct = 0.01
    brick_multiplier = 2

    def init(self):
        # Moon Dev Indicator Initialization 🌕📊
        # Bollinger Bandwidth
        def bbw_calc(close):
            upper, middle, lower = talib.BBANDS(close, 
                timeperiod=self.bb_period, 
                nbdevup=self.bb_dev,
                nbdevdn=self.bb_dev)
            return (upper - lower) / middle
            
        self.bbw = self.I(bbw_calc, self.data.Close, name='BBW')
        
        # ATR for brick size
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, 
                        self.data.Close, self.atr_period, name='ATR')
        
        # Swing levels
        self.swing_high = self.I(talib.MAX, self.data.High, 20, name='Swing_High')
        self.swing_low = self.I(talib.MIN, self.data.Low, 20, name='Swing_Low')
        
        # Renko state tracking
        self.last_brick_close = None
        self.renko_trend = None
        self.prev_trend = None

    def next(self):
        # Moon Dev safety checks 🌙🔒
        if len(self.data) < max(self.bb_period, self.atr_period, 20) + 1:
            return
            
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        current_atr = self.atr[-1]
        
        # Update Renko bricks 🧱🌕
        self.update_renko(current_high, current_low, current_atr)
        
        # Calculate conditions
        bbw_cond = self.check_bbw()
        vol_imbalance = self.calculate_volume_imbalance()
        reversal, direction = self.check_reversal()
        
        # Moon Dev entry logic 🌕🚀
        if all([bbw_cond, abs(vol_imbalance) >= self.volume_threshold, reversal]):
            if direction == 'up' and vol_imbalance > 0:
                self.enter_long(current_close)
            elif direction == 'down' and vol_imbalance < 0:
                self.enter_short(current_close)

    def update_renko(self, high, low, atr):
        brick_size = atr * self.brick_multiplier
        if self.last_brick_close is None:
            self.last_brick_close = self.data.Close[0]
            return
            
        if self.renko_trend == 'up':
            if high >= self.last_brick_close + brick_size:
                self.last_brick_close += brick_size
                print(f"🌕 New UP Brick: {self.last_brick_close}")
            elif low <= self.last_brick_close - 2*brick_size:
                self.last_brick_close -= brick_size
                self.renko_trend = 'down'
                print(f"🌑 Reversal DOWN: {self.last_brick_close}")
        elif self.renko_trend == 'down':
            if low <= self.last_brick_close - brick_size:
                self.last_brick_close -= brick_size
                print(f"🌑 New DOWN Brick: {self.last_brick_close}")
            elif high >= self.last_brick_close + 2*brick_size:
                self.last_brick_close += brick