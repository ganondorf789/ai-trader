I'll fix the code by removing all `backtesting.lib` imports and replacing any crossover/crossunder functions with manual implementations. Here's the corrected version with Moon Dev themed debug prints:

```python
import numpy as np
import pandas as pd
import talib
import pandas_ta
from backtesting import Backtest, Strategy

# Load and preprocess data
data = pd.read_csv(
    "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv",
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

# Rename columns to match backtesting requirements
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolatilitySurgeBreakout(Strategy):
    def init(self):
        # Bollinger Bands (20, 2)
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            lambda close: talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0),
            self.data.Close,
            name=['BB_Upper', 'BB_Middle', 'BB_Lower']
        )
        
        # Keltner Channel (20, 2) using pandas_ta
        self.kc_upper, self.kc_middle, self.kc_lower = self.I(
            lambda high, low, close: pandas_ta.kc(
                high=high, low=low, close=close, length=20, scalar=2, mamode='ema'
            ).values.T,
            self.data.High, self.data.Low, self.data.Close,
            name=['KC_Upper', 'KC_Middle', 'KC_Lower']
        )
        
        # Volume MA (20)
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='Volume_MA')
        
        # ATR (14) for trailing stops
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR14')
        
        # Swing High/Low calculations
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='Swing_Low')
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='Swing_High')
        
        # Bollinger Band Width calculations
        self.bb_width = self.I(lambda upper, lower: upper - lower, self.bb_upper, self.bb_lower, name='BB_Width')
        self.bb_width_min = self.I(talib.MIN, self.bb_width, timeperiod=20, name='BB_Width_Min')
        
        self.trailing_stop = None

    def next(self):
        # Moon Dev status updates
        if len(self.data) % 100 == 0:
            print(f"🌙✨ Moon Dev scanning bar {len(self.data)} | Close: {self.data.Close[-1]:.2f} | BB Width: {self.bb_width[-1]:.2f} | Volume Surge: {self.data.Volume[-1] > self.volume_ma[-1]}")

        if not self.position:
            # Long entry conditions
            bb_squeeze = self.bb_width[-1] <= self.bb_width_min[-1]
            kc_expanding = (self.kc_upper[-1] - self.kc_lower[-1]) > (self.kc_upper[-2] - self.kc_lower[-2])
            volume_surge = self.data.Volume[-1] > self.volume_ma[-1]
            price_breakout = self.data.Close[-1] > self.bb_upper[-1]

            if bb_squeeze and kc_expanding and volume_surge and price_breakout:
                risk_percent = 0.01
                risk_amount = self.equity * risk_percent
                sl_price = self.swing_low[-1]
                risk_per_share = self.data.Close[-1] - sl_price
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    self.buy(size=position_size, sl=sl_price)
                    print