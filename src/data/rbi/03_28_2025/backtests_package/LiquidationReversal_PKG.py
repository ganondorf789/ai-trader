Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
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

class LiquidationReversal(Strategy):
    rsi_period = 224  # 14*16 (4h equivalent)
    bb_period = 20
    bb_std = 2
    swing_window = 50
    squeeze_period = 960  # 10 days in 15m
    risk_pct = 0.01
    position_size = 1000000  # Initial equity

    def init(self):
        # Core indicators
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period)
        
        # Bollinger Bands
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, self.bb_period, self.bb_std, self.bb_std, 0)[0]
        self.bb_mid = self.I(talib.BBANDS, self.data.Close, self.bb_period, self.bb_std, self.bb_std, 0)[1]
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, self.bb_period, self.bb_std, self.bb_std, 0)[2]
        self.bb_width = self.I(lambda: (self.bb_upper - self.bb_lower)/self.bb_mid)
        self.bb_low = self.I(talib.MIN, self.bb_width, self.squeeze_period)
        
        # Swing levels
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_window)

    def next(self):
        if len(self.data) < 1000:  # Warmup period
            return

        price = self.data.Close[-1]
        swing_high = self.swing_high[-1]
        swing_low = self.swing_low[-1]

        # Volatility squeeze condition 🌙
        squeeze_on = self.bb_width[-1] <= self.bb_low[-1]

        # RSI divergence detection ✨
        bearish_div = False
        bullish_div = False
        
        # Simplified divergence check (practical implementation needs more sophistication)
        if self.rsi[-1] < np.nanmax(self.rsi[-50:-25]) and price > np.nanmax(self.data.High[-50:-25]):
            bearish_div = True
            print(f"🌙 Bearish divergence detected! RSI weakening at moon high {price:.2f}")
            
        if self.rsi[-1] > np.nanmin(self.rsi[-50:-25]) and price < np.nanmin(self.data.Low[-50:-25]):
            bullish_div = True
            print(f"🌙 Bullish divergence detected! RSI strength at moon low {price:.2f}")

        # Entry logic 🚀
        if not self.position:
            # Short entry
            if (bearish_div and squeeze_on and 
                price > swing_high*0.995 and 
                (self.data.Close[-3] > self.swing_low[-3] and self.data.Close[-1] < self.swing_low[-1])):
                
                sl = swing_high * 1.005
                risk = abs(sl - price)
                position_size = int(round((self.equity * self.risk_pct) / risk))
                if position_size > 0:
                    self.sell(size=position_size, sl=sl, tp=price - 2*risk)
                    print(f"🚀🌑 MOON SHORT! Entry: {price:.2f} | Size: {position_size}")

            # Long entry        
            elif (bullish_div and squeeze_on and
                  price < swing_low*1.005 and
                  (self.data.Close[-