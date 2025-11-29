I'll complete and debug the VolatilitySurge strategy code while maintaining all the original logic and adding Moon Dev themed improvements. Here's the fixed version:

```python
# 🌙 Moon Dev's VolatilitySurge Backtest Implementation ✨
from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import talib

# =====================
# DATA PREPARATION 🌙✨
# =====================
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and prepare data according to Moon Dev specs
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# =====================
# STRATEGY CLASS 🚀🌙
# =====================
class VolatilitySurge(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    atr_period = 20
    atr_avg_period = 90
    volume_lookback = 50
    
    def init(self):
        # 🌙 Core Indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.atr_avg = self.I(talib.SMA, self.atr, timeperiod=self.atr_avg_period)
        
        # 🌙 Volume threshold using pandas rolling quantile
        def volume_thresh(x):
            return x.rolling(self.volume_lookback).quantile(0.95)
        self.volume_thresh = self.I(volume_thresh, self.data.Volume)
        
        # 🌙 Swing high/low indicators
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20)
        
        # 🌙 Trailing stop tracking
        self.trail_high = None
        self.trail_low = None

    def next(self):
        # Skip initial warmup period
        if len(self.data) < max(self.atr_avg_period, self.volume_lookback):
            return
        
        # 🌙✨ Exit Logic: Trailing Stops
        if self.position.is_long:
            self.trail_high = max(self.trail_high, self.data.High[-1]) if self.trail_high else self.data.High[-1]
            stop = self.trail_high - 2 * self.atr[-1]
            if self.data.Low[-1] <= stop:
                self.position.close()
                print(f"🌙🛑 MOON EXIT LONG | Price: {self.data.Close[-1]:.2f} | Trail: {stop:.2f}")
                
        elif self.position.is_short:
            self.trail_low = min(self.trail_low, self.data.Low[-1]) if self.trail_low else self.data.Low[-1]
            stop = self.trail_low + 2 * self.atr[-1]
            if self.data.High[-1] >= stop:
                self.position.close()
                print(f"🌙🛑 MOON EXIT SHORT | Price: {self.data.Close[-1]:.2f} | Trail: {stop:.2f}")

        # 🌙✨ Entry Logic
        if not self.position:
            valid_indicators = all([
                not np.isnan(self.atr[-1]),
                not np.isnan(self.atr_avg[-1]),
                not np.isnan(self.volume_thresh[-1])
            ])
            
            if valid_indicators:
                # Long Entry 🌙🚀
                if (self.data.Close[-1] > self.swing_high[-1] 
                    and self.atr[-1] > self.atr_avg[-1]
                    and self.data.Volume[-1] > self.volume_thresh[-1]):
                    
                    risk_amount = self.broker.equity * self.risk_pct
                    risk_per_share = 2 * self.atr[-1]
                    size = int(round(risk_amount / risk_per_share))
                    
                    if size > 0:
                        self.buy(size=size)
                        self.trail_high = self.data.High