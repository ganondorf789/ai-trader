Here's the implementation of the VortexBreakout strategy for backtesting.py:

```python
# 🌙 Moon Dev's VortexBreakout Backtest 🌙
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib
import pandas_ta as ta
import numpy as np

# ========= DATA PREPROCESSING =========
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names 🧹
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to proper column names 📌
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index ⏰
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

# ========= STRATEGY CLASS =========
class VortexBreakout(Strategy):
    vi_period = 14
    ci_period = 14
    swing_period = 20
    ci_threshold = 38.2
    risk_percent = 0.01  # 1% per trade
    
    def init(self):
        # ========= INDICATOR CALCULATIONS =========
        # Vortex Indicator 🌪️
        vortex = self.data.ta.vortex(
            high=self.data.High,
            low=self.data.Low,
            close=self.data.Close,
            length=self.vi_period
        )
        self.vi_plus = vortex[f'VORTICSP_{self.vi_period}']
        self.vi_minus = vortex[f'VORTICSN_{self.vi_period}']
        
        # Choppiness Index 🌀
        self.ci = self.data.ta.chop(
            high=self.data.High,
            low=self.data.Low,
            close=self.data.Close,
            length=self.ci_period
        )
        
        # Volume SMA 📊
        self.volume_sma = self.I(talib.SMA, self.data.Volume, self.swing_period)
        
        # Swing Highs/Lows 🏔️
        self.recent_high = self.I(talib.MAX, self.data.High, self.swing_period)
        self.recent_low = self.I(talib.MIN, self.data.Low, self.swing_period)
        
        # Add indicator plots 📈
        self.I(lambda: self.vi_plus, name='VI+')
        self.I(lambda: self.vi_minus, name='VI-')
        self.I(lambda: self.ci, name='Choppiness')

    def next(self):
        # ========= ENTRY LOGIC =========
        if not self.position:
            # Get current values 📌
            idx = self._index
            vi_plus = self.vi_plus[idx]
            vi_minus = self.vi_minus[idx]
            ci_value = self.ci[idx]
            current_close = self.data.Close[-1]
            current_volume = self.data.Volume[-1]
            
            # Volume SMA check 📈
            vol_sma = self.volume_sma[-1] if len(self.volume_sma) > 0 else 0
            
            # Swing levels 🏔️
            swing_high = self.recent_high[-1] if len(self.recent_high) > 0 else np.inf
            swing_low = self.recent_low[-1] if len(self.recent_low) > 0 else 0
            
            # Long Entry Conditions 🚀
            if (vi_plus > vi_minus and 
                ci_value < self.ci_threshold and
                current_volume > vol_sma and
                current_close > swing_high):
                
                # Risk management 💰
                risk_amount = self.equity * self.risk_percent
                stop_price = swing_low
                risk_per_share = current_close - stop_price
                
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    if size > 0:
                        self.buy(size=size, sl=stop_price)
                        print(f"🌙✨🚀 LONG ENTRY | Price: {current_close:.2f} | Size: {size} | SL: {stop_price:.2f}")
            
            # Short Entry Conditions 🌧️
            elif (vi_minus > vi_plus and