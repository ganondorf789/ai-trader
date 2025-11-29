I'll complete and fix the code while maintaining all the strategy logic and adding Moon Dev themed debug prints. Here's the fully corrected version:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# Moon Dev's Data Preparation Ritual 🌙✨
def prepare_data(path):
    data = pd.read_csv(path)
    # Cleanse column names
    data.columns = data.columns.str.strip().str.lower()
    # Banish unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    # Align with cosmic conventions
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    # Convert to sacred timeline
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

# Cosmic Configuration 🌌
DATA_PATH = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'

class BandWickSurge(Strategy):
    risk_percent = 0.01  # 1% cosmic risk allocation 🌙
    max_bars_held = 5    # Planetary orbital period 🪐

    def init(self):
        # Celestial Indicators Calculation 🌠
        self.upper, self.middle, self.lower = self.I(
            talib.BBANDS, self.data.Close, 
            timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        
        # Volume Singularity Detector 📈
        self.volume_90th = self.I(
            ta.pctile, self.data.Volume, 
            length=20, percentile=90
        )
        
        # Wick Resonance Scanner 🕯️
        self.wick_signal = self.I(
            self._calculate_wick_strength,
            self.data.Open, self.data.Close, self.data.Low,
            name='Wick Power'
        )

    def _calculate_wick_strength(self, open_, close, low):
        """Calculate lunar wick strength pattern"""
        body = np.abs(close - open_)
        lower_wick = np.minimum(open_, close) - low
        return (lower_wick >= 2 * body).astype(int)

    def next(self):
        current_low = self.data.Low[-1]
        current_close = self.data.Close[-1]
        
        # Cosmic Entry Alignment 🌙✨🚀
        if not self.position:
            entry_conditions = [
                current_low <= self.lower[-1],          # Band contact
                self.data.Volume[-1] >= self.volume_90th[-1],  # Volume surge
                self.wick_signal[-1] == 1               # Wick confirmation
            ]
            
            if all(entry_conditions):
                # Calculate stardust allocation 💫
                equity = self.broker.get_value()
                risk_amount = equity * self.risk_percent
                entry_price = current_close
                stop_price = current_low  # Sacred protection level
                
                risk_per_unit = entry_price - stop_price
                if risk_per_unit <= 0:
                    print("🌑 WARNING: Black hole scenario detected - avoiding trade!")
                    return  # Avoid black hole scenario 🌑
                
                position_size = int(round(risk_amount / risk_per_unit))
                if position_size > 0:
                    self.buy(
                        size=position_size,
                        sl=stop_price,
                        tag="Lunar Surge Activation 🌙✨"
                    )
                    self.entry_bar = len(self.data) - 1
                    print(f"🌙🚀 MOON DEV ENTRY: {position_size} units @ {entry_price} | Shield @ {stop_price} 🌌")

        # Galactic Exit Protocols 🛸
        else:
            # Middle Band Ascension Exit
            if (self.data.Close[-2] < self.middle[-2] and 
                self.data.Close[-1] > self.middle[-1]):
                self.position.close()
                print(f"🌕✅ COSMIC EXIT: Middle band transcendence @ {current_close}")
            
            # Time Vortex Exit (5-bar expiration)
            elif (len(self.data) - 1 - self.entry_bar) >= self.max_bars_held:
                self.position.close()
                print(f"⏳🌀 TEMPORAL EXIT: Maximum orbital period reached @ {current_close}")

#