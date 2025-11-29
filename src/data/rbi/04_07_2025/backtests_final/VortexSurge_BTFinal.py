Here's the complete fixed code with Moon Dev themed debug prints and all technical issues resolved:

```python
import numpy as np
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# Moon Dev data preparation 🌙
print("🌙✨ Initializing Moon Dev's VortexSurge Strategy...")
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

# Clean and format data columns
print("🌙 Cleaning cosmic data dust...")
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VortexSurge(Strategy):
    vi_period = 14
    volume_lookback = 50 * 96  # 50 days in 15-min intervals
    atr_period = 14
    risk_pct = 0.01  # 1% risk per trade
    sl_multiplier = 2
    tp_multiplier = 1.5

    def init(self):
        # 🌙 Vortex Indicator Calculations
        print("🌙 Calculating cosmic vortex energies...")
        vi = ta.vortex(
            high=self.data.High,
            low=self.data.Low,
            close=self.data.Close,
            length=self.vi_period
        )
        self.vi_plus = self.I(lambda: vi[f'VORTICSP_{self.vi_period}'], name='VI+')
        self.vi_minus = self.I(lambda: vi[f'VORTICSN_{self.vi_period}'], name='VI-')

        # 🚀 Volume Percentile Calculation
        print("🌙 Measuring stellar volume patterns...")
        self.vol_percentile = self.I(
            lambda: self.data.Volume.rolling(self.volume_lookback).quantile(0.9),
            name='VolTop10%'
        )

        # ✨ ATR Calculation
        print("🌙 Computing celestial volatility...")
        self.atr = self.I(talib.ATR,
            self.data.High,
            self.data.Low,
            self.data.Close,
            timeperiod=self.atr_period,
            name='ATR'
        )

    def next(self):
        # First check exit conditions
        for trade in self.trades:
            current_atr = trade.tag['entry_atr']
            entry_price = trade.entry_price
            
            if trade.is_long:
                # Check VI crossover exit
                if self.vi_minus[-2] > self.vi_plus[-2] and self.vi_minus[-1] < self.vi_plus[-1]:
                    trade.close()
                    print(f"🌙✨ Closing LONG position: VI crossover detected")
                # Check dynamic price exit
                elif self.data.Close[-1] < (entry_price - self.tp_multiplier * current_atr):
                    trade.close()
                    print(f"🌙✨ Closing LONG position: Price below 1.5x ATR threshold")

            elif trade.is_short:
                # Check VI crossover exit
                if self.vi_plus[-2] < self.vi_minus[-2] and self.vi_plus[-1] > self.vi_minus[-1]:
                    trade.close()
                    print(f"🌙✨ Closing SHORT position: VI crossover detected")
                # Check dynamic price exit
                elif self.data.Close[-1] > (entry_price + self.tp_multiplier * current_atr):
                    trade.close()
                    print(f"🌙✨ Closing SHORT position: Price above 1.5x ATR threshold")

        # Only enter new trades if no existing position
        if not self.position:
            current_vol = self.data.Volume[-1]
            vol_threshold = self.vol_percentile[-1]
            
            # 🌙✨ Long Entry Condition
            if (self.vi_plus[-2] < self.vi_minus[-2] and self.vi_plus[-1] > self.vi_minus[-1]) and current_vol >= vol_threshold:
                atr_value = self.atr[-1]
                risk_amount = self.equity * self.risk_pct
                position_size = int(round(risk_amount / (self.sl_multiplier * atr_value)))