Here's the complete backtesting implementation for the BandwidthContinuation strategy:

```python
# 🌙 Moon Dev's Bandwidth Continuation Backtest 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Data preparation with Moon Dev magic ✨
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and prepare cosmic data 🌌
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class BandwidthContinuation(Strategy):
    def init(self):
        # 🌟 Celestial Indicators 🌟
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, 14)
        
        # 🌗 Bollinger Bands with Moon Phases
        self.upper_band, self.middle_band, self.lower_band = self.I(
            talib.BBANDS, self.data.Close, 
            timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        
        # 🌌 Bandwidth Calculation
        self.bandwidth = self.I(lambda u, l: u - l, self.upper_band, self.lower_band)
        self.bandwidth_sma = self.I(talib.SMA, self.bandwidth, 20)
        self.bandwidth_std = self.I(talib.STDDEV, self.bandwidth, 20)
        
        # 🌑 Swing Points for Lunar Gravity (SL)
        self.swing_high = self.I(talib.MAX, self.data.High, 20)
        self.swing_low = self.I(talib.MIN, self.data.Low, 20)
        
        print("🌙✨ Strategy initialized with Moon Dev cosmic energy!")

    def next(self):
        # 🌕 Moon Phase Debugging
        if len(self.data) % 100 == 0:
            print(f"🌙 Processing bar {len(self.data)} | Price: {self.data.Close[-1]:.2f} | ADX: {self.adx[-1]:.2f}")
        
        # 🚀 Launchpad Entry Logic
        if not self.position:
            current_adx = self.adx[-1]
            prev_adx = self.adx[-2]
            
            # Long Entry Constellation 🌠
            long_conditions = [
                current_adx > 30,
                current_adx > prev_adx,
                self.bandwidth[-1] < self.bandwidth_sma[-1],
                self.data.Close[-2] > self.upper_band[-2],
                self.data.Low[-1] <= self.upper_band[-1],
                self.data.Close[-1] > self.upper_band[-1]
            ]
            
            # Short Entry Constellation 🌌
            short_conditions = [
                current_adx > 30,
                current_adx > prev_adx,
                self.bandwidth[-1] < self.bandwidth_sma[-1],
                self.data.Close[-2] < self.lower_band[-2],
                self.data.High[-1] >= self.lower_band[-1],
                self.data.Close[-1] < self.lower_band[-1]
            ]
            
            # 🌙 Execute Lunar Trades
            if all(long_conditions):
                sl = self.swing_low[-1]
                risk = self.data.Close[-1] - sl
                if risk > 0:
                    size = int(round((self.equity * 0.01) / risk))
                    self.buy(size=size, sl=sl)
                    print(f"🌙🚀 LONG LAUNCH! | Price: {self.data.Close[-1]:.2f} | Size: {size} | Cosmic SL: {sl:.2f}")
            
            elif all(short_conditions):
                sl = self.swing_high[-1]
                risk = sl - self.data.Close[-1]
                if risk > 0:
                    size = int(round((self.equity * 0.01) / risk))
                    self.sell(size=size, sl=sl)
                    print(f"🌙🚀 SHORT LAUNCH! | Price: {self.data.Close[-1]:.2f} | Size