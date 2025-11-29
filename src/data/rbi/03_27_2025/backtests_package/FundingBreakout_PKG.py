I'll fix the code by removing all backtesting.lib imports and replacing any crossover functions with proper implementations. Here's the corrected version with Moon Dev themed debug prints:

```python
# 🌙 Moon Dev Backtesting Engine: FundingBreakout Strategy Implementation ✨

import pandas as pd
import talib
from backtesting import Backtest, Strategy
from datetime import time

# =====================
# DATA PREPARATION 🌌
# =====================
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"

# Load and clean data
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Column mapping with cosmic alignment ✨
column_map = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}
data.rename(columns=column_map, inplace=True, errors='ignore')

# =====================
# CELESTIAL STRATEGY 🌠
# =====================
class FundingBreakout(Strategy):
    time_filter = (time(0,0), time(4,0))  # UTC trading window
    risk_pct = 0.01  # 1% risk per trade
    max_exposure = 0.05  # 5% portfolio limit
    
    def init(self):
        # Cosmic indicators 🌗
        self.funding_low = self.I(talib.MIN, self.data['funding_rate'], timeperiod=30*96)  # 30-day low
        self.n_day_high = self.I(talib.MAX, self.data.High, timeperiod=10*96)  # 10-day high
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14*96)  # 14-day ATR
        
        # Stellar trackers 🌟
        self.entry_price = None
        self.peak_price = None
        self.stop_loss = None

    def next(self):
        current_time = self.data.index[-1].time()
        
        # Only trade during celestial alignment 🌘
        if not (self.time_filter[0] <= current_time < self.time_filter[1]):
            print("🌑 MOON PHASE ALERT: Outside trading hours - skipping")
            return

        # 🌕 ENTRY CONSTELLATION
        if not self.position:
            if (self.data['funding_rate'][-1] <= self.funding_low[-1] and
                self.data.Close[-1] > self.n_day_high[-1]):
                
                # Calculate cosmic position size 🌍
                risk_amount = self.risk_pct * self.equity
                atr_value = self.atr[-1]
                stop_distance = 2 * atr_value
                position_size = risk_amount / stop_distance
                
                # Apply galactic exposure limits 🪐
                current_exposure = sum(p.size * p.entry_price for p in self.positions)
                max_size = (self.max_exposure * self.equity - current_exposure) / self.data.Close[-1]
                position_size = min(position_size, max_size)
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    self.entry_price = self.data.Close[-1]
                    self.peak_price = self.entry_price
                    self.stop_loss = self.entry_price - stop_distance
                    print(f"🚀 MOON DEV LAUNCH: Long {position_size} units at {self.entry_price:.2f}!")
                    print(f"   🌑 Initial Stop: {self.stop_loss:.2f} | ✨ ATR: {atr_value:.2f}")
                else:
                    print("🌘 ORBIT WARNING: Position size too small - skipping trade")

        # 🌑 EXIT GALAXY
        else:
            # Update stellar peak 🌟
            self.peak_price = max(self.peak_price, self.data.High[-1])
            
            # Profit-taking supernova 🌌
            drawdown = (self.peak_price - self.data.Close[-1]) / (self.peak_price - self.entry_price)
            if drawdown >= 0.5:
                self.sell(size=int(round(self.position.size * 0.5)))
                print(f"🌕 HALF-MOON PROFIT: Took 50% at {self.data.Close[-