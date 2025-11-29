Here's the debugged version of your VolatilityBreakout strategy with Moon Dev improvements 🌙✨:

```python
import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Moon Dev Data Preparation Ritual 🌙✨
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Cleanse column names with lunar magic
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Align columns with cosmic order 🌌
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolatilityBreakout(Strategy):
    ema50_period = 50
    ema200_period = 200
    atr_period = 14
    atr_median_window = 20  # For low volatility filter
    risk_pct = 0.01  # 1% risk per trade
    
    def init(self):
        # Cosmic Indicator Calculations 🌠
        self.ema50 = self.I(talib.EMA, self.data.Close, self.ema50_period)
        self.ema200 = self.I(talib.EMA, self.data.Close, self.ema200_period)
        
        # ATR with stellar volatility measurement 🌟
        self.atr = self.I(talib.ATR, 
                         self.data.High, self.data.Low, self.data.Close, 
                         self.atr_period)
        
        # ATR median for low volatility detection 🌗
        self.atr_median = self.I(lambda x: talib.MEDPRICE(x, timeperiod=self.atr_median_window), 
                               self.atr)

    def next(self):
        # Wait for cosmic alignment (enough data) 🌑
        if len(self.data) < 200 or np.isnan(self.atr[-1]) or np.isnan(self.atr_median[-1]):
            return

        # Moon Dev Signal Detection Protocol 🛰️
        atr_low_vol = self.atr[-1] < self.atr_median[-1]
        golden_cross = (self.ema50[-2] < self.ema200[-2]) and (self.ema50[-1] > self.ema200[-1])
        death_cross = (self.ema200[-2] < self.ema50[-2]) and (self.ema200[-1] > self.ema50[-1])

        # Celestial Position Sizing Calculator 🌕
        def calculate_size():
            risk_amount = self.equity * self.risk_pct
            risk_per_share = 2 * self.atr[-1]
            if risk_per_share <= 0: 
                return 0
            return int(round(risk_amount / risk_per_share))  # Ensuring whole number units

        # Long Entry: Galactic Uptrend Initiated 🚀
        if golden_cross and atr_low_vol and not self.position:
            size = calculate_size()
            if size > 0:
                price = self.data.Close[-1]
                sl = price - 2 * self.atr[-1]
                tp = price + 1.5 * self.atr[-1]
                self.buy(size=size, sl=sl, tp=tp)
                print(f"🌙✨ MOON DEV LONG SIGNAL 🌟 Size: {size} | Entry: {price:.2f} | Cosmic SL: {sl:.2f} | Stellar TP: {tp:.2f}")

        # Short Entry: Black Hole Downtrend Detected 🕳️
        elif death_cross and atr_low_vol and not self.position:
            size = calculate_size()
            if size > 0:
                price = self.data.Close[-1]
                sl = price + 2 * self.atr[-1]
                tp = price - 1.5 * self.atr[-1]
                self.sell(size=size, sl=sl, tp=tp)
                print(f"🌙✨ MOON DEV SHORT SIGNAL 🌌 Size: {size} | Entry: {price:.2f} | Event Horizon SL: {sl:.2f} | Singularity TP: {tp:.2f}")

# Launch Backtest Rocket 🚀
bt = Backtest(data, VolatilityBreakout, commission=.002, exclusive_orders=True)
stats