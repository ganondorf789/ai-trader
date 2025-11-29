Here's the fully debugged and Moon Dev themed backtest code with all technical issues fixed while preserving the original strategy logic:

```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
import numpy as np

# 🌙 Lunar Data Loading Sequence
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
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

# 🌌 Custom Indicator Functions
def crossover(series1, series2):
    return (series1[-2] < series2[-2]) & (series1[-1] > series2[-1])

def crossunder(series1, series2):
    return (series1[-2] > series2[-2]) & (series1[-1] < series2[-1])

class FractalVolatilityStrategy(Strategy):
    def init(self):
        # 🌙 Volatility Indicators
        upper, middle, lower = talib.BBANDS(self.data.Close, 20, 2, 2)
        self.bb_upper = self.I(lambda: upper, name='BB_UPPER')
        self.bb_middle = self.I(lambda: middle, name='BB_MID')
        self.bb_lower = self.I(lambda: lower, name='BB_LOWER')
        
        bb_width = (upper - lower)/middle
        self.bb_width_pct = self.I(ta.percentrank, bb_width, 100, name='BB_PCT')
        
        # 🚀 Trend Indicators
        self.vwap = self.I(ta.vwap, self.data.High, self.data.Low, self.data.Close, 
                          self.data.Volume, anchor='D', name='VWAP')
        self.vwap_slope = self.I(talib.LINEARREG_SLOPE, self.vwap, 5, name='VWAP_SLOPE')
        
        # 🌌 Fractal Dimension
        self.hurst = self.I(ta.hurst, self.data.Close, 100, name='HURST')
        self.fractal_dim = self.I(lambda h: 2 - h, self.hurst, name='FRACTAL_DIM')

    def next(self):
        # 🌙 Lunar Status Update
        if len(self.data) % 100 == 0:
            print(f"🌕 Moon Dev Pulse | Bar: {len(self.data)} | BB%: {self.bb_width_pct[-1]:.1f} | "
                  f"VWAP Slope: {self.vwap_slope[-1]:.4f} | Fractal: {self.fractal_dim[-1]:.2f}")

        # 🛑 Risk Management First
        equity = self.equity
        max_risk = 0.01 * equity
        
        if not self.position:
            # 🌙 Long Entry Constellation
            if (self.bb_width_pct[-1] < 20 and
                self.vwap_slope[-1] > 0 and
                crossover(self.data.Close, self.bb_upper)):
                
                entry_price = self.data.Close[-1]
                sl_price = entry_price * 0.98  # 2% stop loss
                risk_per_share = entry_price - sl_price
                position_size = (max_risk / risk_per_share) * (2 - self.fractal_dim[-1])
                position_size = int(round(position_size))  # 🌙 Ensure whole units
                
                if position_size > 0:
                    self.buy(size=position_size, sl=sl_price, 
                           tag=f"🌙 LONG | FD:{self.fractal_dim[-1]:.2f}")
                    print(f"🚀🌙 MOON LAUNCH! LONG {position_size} units @ {entry_price:.2f} | "
                          f"SL: {sl_price:.2f} 🌌")

            # 🌑 Short Entry Eclipse
            elif (self.bb_width_pct[-1] < 20 and
                  self.vwap_slope[-1] < 0 and
                  crossunder(self.data.Close, self.bb_lower)):
                  
                entry_price = self.data.Close[-1]
                sl_price = entry_price *