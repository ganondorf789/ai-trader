I notice the code was cut off at the short exit condition. Let me complete and debug the code while maintaining all the original strategy logic. Here's the fixed version with Moon Dev enhancements:

```python
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

# Load and prepare lunar data 🌙
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolatilitySurge(Strategy):
    ema_period = 20
    atr_period = 20
    atr_multiplier = 2.0
    risk_pct = 0.01  # Risk 1% per trade 🌕

    def init(self):
        # Moon Dev Indicators 🚀
        self.ema20 = self.I(talib.EMA, self.data.Close, self.ema_period)
        self.atr20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.upper_band = self.I(lambda ema, atr: ema + self.atr_multiplier * atr, 
                                self.ema20, self.atr20, name='UPPER BAND')
        self.lower_band = self.I(lambda ema, atr: ema - self.atr_multiplier * atr,
                                self.ema20, self.atr20, name='LOWER BAND')
        
        self.entry_price = None
        self.max_price = -np.inf
        self.min_price = np.inf

    def next(self):
        current_close = self.data.Close[-1]
        prev_close = self.data.Close[-2] if len(self.data.Close) >= 2 else current_close

        # Moon Dev Trading Logic 🌑➡️🌕
        if not self.position:
            # Long Entry Constellation 🌠
            if current_close > self.upper_band[-1] and prev_close <= self.upper_band[-2]:
                sl = self.lower_band[-1]
                risk_per_share = current_close - sl
                
                if risk_per_share > 0:
                    size = int(round((self.equity * self.risk_pct) / risk_per_share))
                    if size > 0:
                        print(f"🚀🌕 LUNAR LIFT-OFF! Long {size} units at {current_close:.2f}")
                        self.buy(size=size)
                        self.entry_price = current_close
                        self.max_price = current_close

            # Short Entry Meteor Shower 🌠
            elif current_close < self.lower_band[-1] and prev_close >= self.lower_band[-2]:
                sl = self.upper_band[-1]
                risk_per_share = sl - current_close
                
                if risk_per_share > 0:
                    size = int(round((self.equity * self.risk_pct) / risk_per_share))
                    if size > 0:
                        print(f"🌑🌘 COMET DIVE! Short {size} units at {current_close:.2f}")
                        self.sell(size=size)
                        self.entry_price = current_close
                        self.min_price = current_close
        else:
            # Moon Trajectory Updates 🌓
            if self.position.is_long:
                self.max_price = max(self.max_price, self.data.High[-1])
                exit_price = self.max_price - (self.max_price - self.entry_price) * 0.5
                
                if current_close < exit_price:
                    print(f"🌕💫 LUNAR DESCENT! Closing long at {current_close:.2f}")
                    self.position.close()
                    self.entry_price = None
                    self.max_price = -np.inf

            elif self.position.is_short:
                self.min_price = min(self.min_price, self.data.Low[-1])
                exit_price = self.min_price + (self.entry_price - self.min_price) * 0.5
                
                if current_close > exit_price:
                    print(f"🌑✨ COMET RECOVERY! Closing short at {current_close:.2f}")
                    self.position.close()
                    self.entry_price = None
                    self.min