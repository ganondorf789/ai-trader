I'll analyze the code and ensure it's free from any `backtesting.lib` imports or functions. Here's the fixed version with Moon Dev themed improvements:

```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

# Data handling
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

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

class LiquidationSqueeze(Strategy):
    bb_period = 20
    bb_dev = 2
    swing_period = 50
    volume_ma_period = 20
    risk_pct = 0.01
    tp_multiplier = 2
    squeeze_lookback = 50

    def init(self):
        # Bollinger Bands components
        self.bb_upper = self.I(self._bb_upper)
        self.bb_middle = self.I(self._bb_middle)
        self.bb_lower = self.I(self._bb_lower)
        
        # Liquidation clusters proxy
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period)
        
        # Volume confirmation
        self.volume_ma = self.I(talib.SMA, self.data.Volume, self.volume_ma_period)
        
        # Squeeze detection tracking
        self.bandwidth_history = []

    def _bb_upper(self, close):
        upper, _, _ = talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)
        return upper

    def _bb_middle(self, close):
        _, middle, _ = talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)
        return middle

    def _bb_lower(self, close):
        _, _, lower = talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)
        return lower

    def next(self):
        if len(self.data) < max(self.bb_period, self.swing_period, self.squeeze_lookback) + 1:
            return

        # Calculate current bandwidth
        current_upper = self.bb_upper[-1]
        current_lower = self.bb_lower[-1]
        current_middle = self.bb_middle[-1]
        bandwidth = (current_upper - current_lower) / current_middle
        
        # Update bandwidth history
        self.bandwidth_history.append(bandwidth)
        if len(self.bandwidth_history) > self.squeeze_lookback:
            self.bandwidth_history.pop(0)
        
        # Check squeeze condition
        is_squeeze = False
        if len(self.bandwidth_history) >= self.squeeze_lookback:
            min_bandwidth = min(self.bandwidth_history)
            is_squeeze = bandwidth <= min_bandwidth

        if is_squeeze and not self.position:
            current_close = self.data.Close[-1]
            current_volume = self.data.Volume[-1]
            volume_ma = self.volume_ma[-1]
            
            # Long entry logic
            if current_close > current_upper and current_volume > volume_ma:
                swing_high = self.swing_high[-1]
                if current_close >= swing_high * 0.99:
                    sl_price = current_lower
                    risk_per_share = current_close - sl_price
                    if risk_per_share > 0:
                        risk_amount = self.equity * self.risk_pct
                        size = int(round(risk_amount / risk_per_share))
                        if size > 0:
                            tp_price = current_close + (current_close - sl_price) * self.tp_multiplier
                            self.buy(size=size, sl=sl_price, tp=tp_price)
                            print(f"🌙✨ MOON LIFT-OFF! LONG {size} units at {current_close} 🚀")
                            print(f"🌌 Stellar stop at {sl_price} | Target orbit {tp_price} 🌠")