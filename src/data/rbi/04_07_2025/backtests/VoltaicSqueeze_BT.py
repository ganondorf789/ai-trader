Here's the complete backtesting implementation for the VoltaicSqueeze strategy:

```python
# 🌙 Moon Dev's VoltaicSqueeze Backtest Implementation 🚀

import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Data Preparation 🌌
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and prepare data columns
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

class VoltaicSqueeze(Strategy):
    # Strategy Parameters ⚙️
    risk_pct = 0.01
    atr_multiplier = 1.5
    exit_bars = 10
    volume_filter = True
    
    def init(self):
        # Core Indicators Calculation 🌗
        self.upper_band, self.middle_band, self.lower_band = self.I(
            talib.BBANDS, self.data.Close, 
            timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        self.atr = self.I(talib.ATR, 
            self.data.High, self.data.Low, self.data.Close, 20
        )
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20)
        
        print("🌙 Moon Dev Indicators Initialized: BBANDS, ATR, Volume SMA ✨")

    def next(self):
        # Wait for sufficient data 🌊
        if len(self.data) < 40:
            return

        # Previous bar values 📊
        prev_idx = -2
        prev_close = self.data.Close[prev_idx]
        prev_upper = self.upper_band[prev_idx]
        prev_atr = self.atr[prev_idx]
        prev_volume = self.data.Volume[prev_idx]
        prev_volume_sma = self.volume_sma[prev_idx]

        # Entry Conditions Check 🔍
        if not self.position:
            bb_width = prev_upper - self.lower_band[prev_idx]
            volatility_ok = prev_atr > prev_close * 0.005
            volume_ok = prev_volume > prev_volume_sma if self.volume_filter else True
            
            if (prev_close > prev_upper and
                bb_width < (prev_atr * self.atr_multiplier) and
                volatility_ok and
                volume_ok):
                
                # Risk Management Calculation 🔒
                entry_price = self.data.Open[-1]
                breakout_range = self.data.High[prev_idx] - self.data.Low[prev_idx]
                initial_stop = entry_price - (breakout_range * 0.5)
                risk_amount = self.equity * self.risk_pct
                stop_distance = entry_price - initial_stop
                
                if stop_distance <= 0:
                    print("🌑 Invalid stop distance - Trade aborted")
                    return
                
                position_size = int(round(risk_amount / stop_distance))
                
                if position_size > 0:
                    self.buy(size=position_size, 
                           tag=f"VoltaicSqueeze Entry 🌪️")
                    print(f"🚀 LONG Entry: {entry_price:.2f} | Size: {position_size}")
                    
                    # Initialize exit parameters 📉
                    self.trailing_stop = initial_stop
                    self.highest_high = entry_price
                    self.bars_since_high = 0

        # Exit Management 🛑
        else:
            # Update trailing stop 🌊
            current_high = self.data.High[-1]
            current_range = current_high - self.data.Low[-1]
            new_stop = current_high - (current_range * 0.25)
            self.trailing_stop = max(self.trailing_stop, new_stop)
            
            # Update highest high tracking 🌙
            if current_high > self.highest_high:
                self.highest_high = current_high
                self.bars_since_high = 0
            else:
                self.bars_since_high += 1
            
            # Exit Conditions Check 🚨