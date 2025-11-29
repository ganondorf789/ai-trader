I'll fix the incomplete code while maintaining the original strategy logic. Here's the corrected version with Moon Dev themed debug prints:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib as ta
import numpy as np

# Load and prepare data
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class DeltaSurgeReversal(Strategy):
    risk_pct = 0.01
    rsi_period = 14
    swing_window = 20
    volume_lookback = 50
    
    def init(self):
        # Core strategy indicators
        self.rsi = self.I(ta.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.swing_high = self.I(ta.MAX, self.data.High, timeperiod=self.swing_window)
        self.swing_low = self.I(ta.MIN, self.data.Low, timeperiod=self.swing_window)
        self.volume_threshold = self.I(
            lambda x: x.rolling(self.volume_lookback).quantile(0.9),
            self.data.Volume
        )
        print("🌙 MOON DEV INDICATORS INITIALIZED! Ready for lunar trading cycles... ✨")
        
    def next(self):
        price = self.data.Close[-1]
        
        if not self.position:
            # 🌙 Moon Dev Entry Logic
            if self._long_signal():
                self._enter_long()
                
            elif self._short_signal():
                self._enter_short()

    def _long_signal(self):
        signal = (self.rsi[-1] < 30 
                and self.data.Volume[-1] > self.volume_threshold[-1])
        if signal:
            print("🌕 MOON DEV LONG SIGNAL DETECTED! RSI oversold with volume surge...")
        return signal
    
    def _short_signal(self):
        signal = (self.rsi[-1] > 70 
                and self.data.Volume[-1] > self.volume_threshold[-1])
        if signal:
            print("🌘 MOON DEV SHORT SIGNAL DETECTED! RSI overbought with volume surge...")
        return signal
    
    def _enter_long(self):
        swing_low = self.swing_low[-1]
        risk_amount = self.equity * self.risk_pct
        stop_price = swing_low * 0.99  # 1% below swing low
        
        size = int(round(risk_amount / (self.data.Close[-1] - stop_price)))
        if size > 0:
            fib_level = self._calculate_fib_levels(long=True)
            self.buy(size=size, 
                    sl=stop_price,
                    tp=fib_level,
                    tag="🌙 Lunar Long")
            print(f"🚀 MOON DEV LONG LAUNCH! Size: {size} | "
                  f"Entry: {self.data.Close[-1]:.2f} | "
                  f"SL: {stop_price:.2f} | TP: {fib_level:.2f} 🌕")
            print("🌙 Lunar Gravity Assist Activated! Awaiting profit trajectory...")

    def _enter_short(self):
        swing_high = self.swing_high[-1]
        risk_amount = self.equity * self.risk_pct
        stop_price = swing_high * 1.01  # 1% above swing high
        
        size = int(round(risk_amount / (stop_price - self.data.Close[-1])))
        if size > 0:
            fib_level = self._calculate_fib_levels(long=False)
            self.sell(size=size,
                     sl=stop_price,
                     tp=fib_level,
                     tag="🌑 Lunar Short")
            print(f"🌑 MOON DEV SHORT BLAST! Size: {size} | "
                  f"Entry: {self.data.Close[-1]:.2f} | "
                  f"SL: {stop_price:.2f} | TP: {fib_level:.2f} 🌘")
            print("🌑 Dark Side Maneuver Initiated! Preparing for descent...")
    
    def _calculate_fib_levels(self