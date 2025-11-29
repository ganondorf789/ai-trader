I'll fix the code by removing all `backtesting.lib` imports and replacing any crossover/crossunder functions with manual implementations. Here's the corrected version with Moon Dev themed debug prints:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names and drop unnamed
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
})

# Set datetime index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolatilityBandit(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Precompute indicators using TA-Lib
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume
        
        # Bollinger Bands
        bb_upper, _, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        self.I(lambda: bb_upper, name='BB_upper')
        self.I(lambda: bb_lower, name='BB_lower')
        
        # Keltner Channels
        ema = talib.EMA(close, 20)
        atr = talib.ATR(high, low, close, 20)
        kc_upper = ema + atr * 1.5
        kc_lower = ema - atr * 1.5
        self.I(lambda: kc_upper, name='KC_upper')
        self.I(lambda: kc_lower, name='KC_lower')
        
        # Volume SMA
        volume_sma = talib.SMA(volume, 20)
        self.I(lambda: volume_sma, name='Volume_SMA')
        
    def next(self):
        # Moon Dev check: require minimum data
        if len(self.data) < 40:  # Allow buffer for indicator calculations
            print("🌙 Moon Dev Warning: Not enough data yet - waiting for more bars")
            return
        
        prev_idx = -2  # Previous completed bar
        current_idx = -1  # Current developing bar
        
        # Existing position management
        if self.position:
            if self.position.is_long:
                # Check exit conditions
                prev_close = self.data.Close[prev_idx]
                exit_bb = prev_close < self.data['BB_upper'][prev_idx]
                exit_kc = prev_close < self.data['KC_upper'][prev_idx]
                if exit_bb or exit_kc:
                    self.position.close()
                    print(f"🌙 Moon Dev Exit LONG: Closed at {self.data.Close[current_idx]:.2f}")
                else:
                    # Update trailing stop to max of current lower bands
                    new_stop = max(self.data['BB_lower'][current_idx], self.data['KC_lower'][current_idx])
                    self.position.stop = new_stop
                    print(f"✨ Moon Dev Trailing Stop Updated: {new_stop:.2f}")
                    
            elif self.position.is_short:
                prev_close = self.data.Close[prev_idx]
                exit_bb = prev_close > self.data['BB_lower'][prev_idx]
                exit_kc = prev_close > self.data['KC_lower'][prev_idx]
                if exit_bb or exit_kc:
                    self.position.close()
                    print(f"🌙 Moon Dev Exit SHORT: Closed at {self.data.Close[current_idx]:.2f}")
                else:
                    # Update trailing stop to min of current upper bands
                    new_stop = min(self.data['BB_upper'][current_idx], self.data['KC_upper'][current_idx])
                    self.position.stop = new_stop
                    print(f"✨ Moon Dev Trailing Stop Updated: {new_stop:.2f}")
        else:
            # Entry logic
            prev_close = self.data.Close[prev_idx]
            prev_volume = self.data.Volume[prev_idx]
            volume_sma = self.data['Volume_SMA'][prev_idx]
            
            # Long conditions
            above_bb = prev_close > self.data['BB_upper'][prev_idx]
            above_kc = prev_close > self.data['