Here's the complete fixed code with Moon Dev themed improvements and all technical issues resolved:

```python
import pandas as pd
from backtesting import Strategy, Backtest
import talib
import numpy as np

# Load and preprocess data with celestial precision 🌙
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VoltaicSqueeze(Strategy):
    def init(self):
        # Core indicators - powered by Moon Dev's celestial calculations ✨
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=20)
        self.upper = self.I(lambda ema, atr: ema + 2*atr, self.ema, self.atr)
        self.lower = self.I(lambda ema, atr: ema - 2*atr, self.ema, self.atr)
        self.width = self.I(lambda u, l: u - l, self.upper, self.lower)
        self.width_50_avg = self.I(talib.SMA, self.width, timeperiod=50)
        
        # Confirmation indicators
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
        # Divergence detection using Moon Dev's proprietary algorithms
        self.price_highs = self.I(talib.MAX, self.data.High, timeperiod=5)
        self.rsi_highs = self.I(talib.MAX, self.rsi, timeperiod=5)
        self.price_lows = self.I(talib.MIN, self.data.Low, timeperiod=5)
        self.rsi_lows = self.I(talib.MIN, self.rsi, timeperiod=5)
        
        self.trade_params = {}
        print("🌙✨ MOON DEV INIT: Strategy initialized with celestial precision!")

    def next(self):
        price = self.data.Close[-1]
        equity = self.equity
        
        # Moon Dev risk management system
        if len(self.trades) >= 3 and all(trade.pl < 0 for trade in self.trades[-3:]):
            print("🌙🚨 MOON DEV ALERT: 3 consecutive losses! Trading halted for safety.")
            return

        if not self.position:
            # Volatility squeeze condition
            squeeze = (self.width[-1] < 0.3 * self.width_50_avg[-1]) 
            mid_lower = self.lower[-1] + 0.25*self.width[-1]
            mid_upper = self.upper[-1] - 0.25*self.width[-1]
            in_middle = mid_lower < price < mid_upper
            
            # Divergence checks using array indexing
            bull_div = (self.price_lows[-1] < self.price_lows[-2]) and \
                      (self.rsi_lows[-1] > self.rsi_lows[-2])
            bear_div = (self.price_highs[-1] > self.price_highs[-2]) and \
                      (self.rsi_highs[-1] < self.rsi_highs[-2])
            
            # Volume surge detection
            volume_ok = self.data.Volume[-1] > 1.2 * self.volume_sma[-1]
            
            # Long entry constellation
            if squeeze and in_middle and bull_div and price > self.upper[-1] and volume_ok and self.rsi[-1] > 50:
                sl = self.lower[-1]
                risk = price - sl
                if risk <= 0: 
                    print("🌙⚠️ Risk calculation failed - aborting launch sequence")
                    return
                size = int(round((0.01 * equity) / risk))
                if size > 0:
                    self.buy(size=size, sl=sl)
                    print(f"🌙🚀 MOON BLASTOFF! LONG {size} units @ {price:.2f} (