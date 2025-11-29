Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations:

```python
# 🌙 Moon Dev Backtest AI Implementation 🚀
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import os

# Clean and prepare data 🌙
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean column names ✨
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping 🌕
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class LiquiClusterBreakout(Strategy):
    risk_pct = 0.02  # 2% risk per trade 🌙
    cluster_window = 20  # Liquidation cluster period
    volume_ma_period = 10  # Taker volume MA
    spread_ma_period = 5  # Spread analysis window
    
    def init(self):
        # 🌙 Liquidation Cluster Indicators
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.cluster_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.cluster_window)
        
        # ✨ Market Structure Indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.spread = self.I(lambda: self.data.High - self.data.Low)
        self.spread_ma = self.I(talib.SMA, self.spread, self.spread_ma_period)
        self.volume_ma = self.I(talib.SMA, self.data.Volume, self.volume_ma_period)
        self.rsi = self.I(talib.RSI, self.data.Close, 14)  # Order book imbalance proxy

    def next(self):
        # 🌑 Avoid low liquidity periods
        if self.data.index[-1].weekday() >= 5:  # Skip weekends
            return

        price = self.data.Close[-1]
        spread = self.spread[-1]
        spread_ma = self.spread_ma[-1]
        
        # 🌙 Moon Dev Debug Prints ✨
        if len(self.data) % 100 == 0:
            print(f"🌙 Lunar Cycle {len(self.data)}: Price={price:.2f} | Equity={self.equity:,.0f} 🚀")

        # Existing position management
        if self.position:
            # Check spread-based exit
            if spread > spread_ma * 1.2:
                self.position.close()
                print(f"🌙 Exit! Spread Explosion {spread:.2f} > {spread_ma*1.2:.2f} ✨")
            return

        # 🚀 Long Entry Logic
        long_trigger = (price >= self.swing_high[-1] * 0.99 and 
                       self.rsi[-1] > 60 and
                       self.data.Volume[-1] >= 2 * self.volume_ma[-1] and
                       spread <= 0.5 * spread_ma)
        
        # 🌑 Short Entry Logic
        short_trigger = (price <= self.swing_low[-1] * 1.01 and
                        self.rsi[-1] < 40 and
                        self.data.Volume[-1] >= 2 * self.volume_ma[-1] and
                        spread <= 0.5 * spread_ma)

        # Execute trades with proper position sizing 🌕
        if long_trigger:
            sl = self.swing_low[-1]
            risk_per_share = price - sl
            if risk_per_share > 0:
                size = int(round((self.risk_pct * self.equity) / risk_per_share))
                tp = price + 1.5 * self.atr[-1]
                self.buy(size=size, sl=sl, tp=tp)
                print(f"🚀 LUNAR LIFT OFF! Long {size} @ {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f} 🌙")

        elif short_trigger:
            sl = self.swing_high[-1]
            risk_per_share = sl - price