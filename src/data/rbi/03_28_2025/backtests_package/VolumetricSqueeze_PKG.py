I'll fix the code by removing all backtesting.lib imports and replacing the crossover function with the proper implementation. Here's the corrected version with Moon Dev themed debug prints:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta

# Load and preprocess data
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean column names and drop unnamed
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Remap columns to backtesting.py format
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
})

class VolumetricSqueeze(Strategy):
    vwma_period = 50
    kc_period = 14
    kc_multiplier = 1.5
    percentile_lookback = 100
    risk_pct = 0.01
    swing_lookback = 5

    def init(self):
        # Calculate VWMA using TA-Lib SMA
        price_volume = self.data.Close * self.data.Volume
        sma_price_volume = self.I(talib.SMA, price_volume, timeperiod=self.vwma_period)
        sma_volume = self.I(talib.SMA, self.data.Volume, timeperiod=self.vwma_period)
        self.vwma = self.I(lambda x,y: x/y, sma_price_volume, sma_volume, name='VWMA')

        # Calculate Keltner Channel
        self.kc_mid = self.I(talib.EMA, self.data.Close, self.kc_period, name='KC_MID')
        atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.kc_period)
        self.kc_upper = self.I(lambda x,y: x + self.kc_multiplier*y, self.kc_mid, atr, name='KC_UPPER')
        self.kc_lower = self.I(lambda x,y: x - self.kc_multiplier*y, self.kc_mid, atr, name='KC_LOWER')

        # Calculate channel width and percentiles
        self.width = (self.kc_upper - self.kc_lower)/self.kc_mid  # Added missing width definition
        self.width_20 = self.I(ta.percentile, self.width, length=self.percentile_lookback, percentile=20, name='WIDTH_20')
        self.width_80 = self.I(ta.percentile, self.width, length=self.percentile_lookback, percentile=80, name='WIDTH_80')

        # Swing low for stop loss
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_lookback, name='SWING_LOW')

    def next(self):
        # Moon Dev debug prints 🌙
        current_bar = len(self.data)-1
        
        if self.position:
            # Check volatility expansion exit
            if self.width[-1] > self.width_80[-1]:
                self.position.close()
                print(f"🌕💥 MOON EXIT: Volatility expanded! Width {self.width[-1]:.4f} > 80th ({self.width_80[-1]:.4f})")

        else:
            # Entry conditions check - replaced crossover with direct comparison
            vwma_cross = (self.data.Close[-2] < self.vwma[-2] and self.data.Close[-1] > self.vwma[-1])
            width_contraction = self.width[-1] < self.width_20[-1]

            if vwma_cross and width_contraction:
                # Calculate risk-based position size
                stop_price = self.swing_low[-1]
                risk_amount = self.equity * self.risk_pct
                risk_per_share = self.data.Close[-1] - stop_price
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.buy(size=position_size, sl=stop_price)
                        print(f"🌙🚀 MOON ENTRY! Size: {position_size} | Risk: {self.risk