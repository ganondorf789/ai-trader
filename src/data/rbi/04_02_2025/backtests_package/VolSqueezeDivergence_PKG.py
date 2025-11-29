I'll fix the code by removing all `backtesting.lib` imports and replacing any related functions with proper alternatives. Here's the corrected version with Moon Dev enhancements:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Data preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
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

class VolSqueezeDivergence(Strategy):
    risk_per_trade = 0.01
    squeeze_lookback = 120
    atr_period = 20
    swing_lookback = 20
    divergence_length = 3

    def init(self):
        # Bollinger Bands components
        self.sma20 = self.I(talib.SMA, self.data.Close, timeperiod=20)
        self.std20 = self.I(talib.STDDEV, self.data.Close, timeperiod=20)
        self.bb_upper = self.I(lambda sma, std: sma + 2*std, self.sma20, self.std20, name='BB_Upper')
        self.bb_lower = self.I(lambda sma, std: sma - 2*std, self.sma20, self.std20, name='BB_Lower')
        self.bandwidth = self.I(lambda u, l: u - l, self.bb_upper, self.bb_lower, name='Bandwidth')
        self.bandwidth_min = self.I(talib.MIN, self.bandwidth, timeperiod=self.squeeze_lookback, name='Bandwidth_Min')

        # OBV and swing levels
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume, name='OBV')
        self.price_lows = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback, name='Price_Lows')
        self.obv_lows = self.I(talib.MIN, self.obv, timeperiod=self.swing_lookback, name='OBV_Lows')
        self.price_highs = self.I(talib.MAX, self.data.High, timeperiod=self.swing_lookback, name='Price_Highs')
        self.obv_highs = self.I(talib.MAX, self.obv, timeperiod=self.swing_lookback, name='OBV_Highs')

        # Volatility components
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period, name='ATR')

        # Trade tracking
        self.entry_price = None
        self.stop_loss = None
        self.trailing_high = None
        self.trailing_low = None

    def next(self):
        if len(self.data) < max(self.squeeze_lookback, self.swing_lookback) + self.divergence_length:
            return

        # Squeeze detection
        squeeze_active = self.bandwidth[-1] <= self.bandwidth_min[-1]
        moon_phase = "🌑" if squeeze_active else "🌕"
        print(f"{moon_phase} Moon Dev Squeeze Scan: Bandwidth {self.bandwidth[-1]:.2f} vs Min {self.bandwidth_min[-1]:.2f}")

        # Divergence detection
        bullish_div = all(self.price_lows[-i] < self.price_lows[-i-1] and 
                        self.obv_lows[-i] > self.obv_lows[-i-1] 
                        for i in range(1, self.divergence_length))

        bearish_div = all(self.price_highs[-i] > self.price_highs[-i-1] and 
                        self.obv_highs[-i] < self.obv_highs[-i-1] 
                        for i in range(1, self.divergence_length))

        # Entry logic
        if not self.position and squeeze_active:
            if bullish_div:
                risk_amount = self.equity * self.risk_per_trade
                sl_level = np.min