Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed indicators:

```python
# 🌙 Moon Dev's Liquidation Breakout Backtest 🌙
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

# 🚀 Data Loading & Cleaning
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

class LiquidationBreakout(Strategy):
    risk_per_trade = 0.02  # 🌙 2% cosmic risk limit
    kc_ema = 20
    atr_multiplier = 2
    cluster_window = 20
    cluster_pct = 0.015
    atr_period = 14

    def init(self):
        # 🌌 Core Indicators (Pure TA-Lib powered)
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.kc_ema)
        self.atr_kc = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.kc_ema)
        self.kc_upper = self.ema + (self.atr_multiplier * self.atr_kc)
        self.kc_lower = self.ema - (self.atr_multiplier * self.atr_kc)
        self.kc_width = self.kc_upper - self.kc_lower
        self.kc_width_avg = self.I(talib.SMA, self.kc_width, timeperiod=20)
        
        # 🔭 Liquidation Proxies (Cosmic Swing Detection)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.cluster_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.cluster_window)
        
        # 📈 Volatility Measures (Lunar Cycle Adjusted)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.vol_avg = self.I(talib.SMA, self.data.Volume, timeperiod=20)

    def next(self):
        if self.position:
            # 🌠 Exit Conditions (Returning to Earth)
            if self.data.Close[-1] >= self.tp or self.data.Close[-1] <= self.sl:
                print(f"🌙 Closing Moon Position at {self.data.Close[-1]}! Cosmic Profits Achieved! 🚀")
                self.position.close()
        else:
            # 🌗 Liquidation Cluster Check (Gravitational Pull Analysis)
            prev_close = self.data.Close[-2] if len(self.data.Close) > 1 else 0
            sh_prev = self.swing_high[-2]
            sl_prev = self.swing_low[-2]

            # 🌟 Long Conditions (Ascending to the Stars)
            long_cluster = (sh_prev - prev_close)/prev_close <= self.cluster_pct
            long_break = self.data.Close[-1] > sh_prev
            kc_expanded = self.kc_width[-1] > 2*self.kc_width_avg[-1]
            vol_confirm = self.data.Volume[-1] > self.vol_avg[-1]

            if long_cluster and long_break and kc_expanded and vol_confirm:
                risk = self.equity * self.risk_per_trade
                fixed_sl = sl_prev
                dynamic_sl = self.data.Close[-1] - self.atr[-1]
                sl = max(fixed_sl, dynamic_sl)
                risk_size = self.data.Close[-1] - sl
                
                if risk_size > 0:
                    size = int(round(risk / risk_size))
                    if size > 0:
                        self.buy(size=size)
                        self.tp = self.data.Close[-1] + 1.5*(self.data.Close[-1]-sl)
                        self.sl = sl
                        print(f"🚀🌙 LONG! {self.data.Close[-1]} | TP: {self.tp:.1