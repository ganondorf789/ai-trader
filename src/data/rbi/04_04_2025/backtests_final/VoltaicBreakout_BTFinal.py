I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete debugged version with Moon Dev themed improvements:

```python
# 🌙 Moon Dev's VoltaicBreakout Backtest Implementation 🚀
import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
import talib
import os

# Clean and prepare data 🌙
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Data cleaning magic ✨
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

class VoltaicBreakout(Strategy):
    bb_period = 20
    bb_dev = 2
    adx_period = 14
    atr_period = 14
    risk_pct = 0.01  # 1% risk per trade 🌙

    def init(self):
        # TA-Lib Indicators with self.I() ✨
        self.bb_upper = self.I(lambda x: talib.BBANDS(x, self.bb_period, self.bb_dev, self.bb_dev)[0], 
                              self.data.Close)
        self.bb_lower = self.I(lambda x: talib.BBANDS(x, self.bb_period, self.bb_dev, self.bb_dev)[2], 
                              self.data.Close)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, self.adx_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        print("🌙 VoltaicBreakout Strategy Activated! ✨")
        print(f"BB({self.bb_period}/{self.bb_dev}) | ADX({self.adx_period}) | ATR({self.atr_period}) 🚀")

    def next(self):
        # Wait for indicators to warm up 🌙
        if len(self.data.Close) < max(self.bb_period, self.adx_period, self.atr_period) + 2:
            return

        # Current market conditions ✨
        price = self.data.Close[-1]
        prev_price = self.data.Close[-2]
        adx = self.adx[-1]
        prev_adx = self.adx[-2]
        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]
        prev_bb_upper = self.bb_upper[-2]
        prev_bb_lower = self.bb_lower[-2]

        # Core strategy logic 🌙
        if not self.position:
            # Trend strength filter ✨
            if adx < 25:
                return

            # Long entry constellation 🌕
            if (price > bb_upper and 
                adx > 25 and adx > prev_adx and
                prev_price < prev_bb_upper and prev_price > prev_bb_lower):
                
                atr_val = self.atr[-1]
                if atr_val <= 0:
                    return
                
                risk_amount = self.equity * self.risk_pct
                stop_dist = 2 * atr_val
                size = int(round(risk_amount / stop_dist))
                
                if size > 0:
                    self.buy(
                        size=size,
                        sl=price - stop_dist,
                        tag={
                            'stop': price - stop_dist,
                            'trail_stop': price - stop_dist,
                            'peak': price
                        }
                    )
                    print(f"🚀 LONG LAUNCHED! 🌙 Size: {size} | Stop: {price - stop_dist:.2f} ✨")

            # Short entry constellation 🌑
            elif (price < bb_lower and 
                  adx > 25 and adx > prev_adx and
                  prev_price < prev_bb_upper and prev_price > prev_bb_lower):
                
                atr_val = self.atr[-1]
                if atr_val <= 0:
                    return
                
                risk_amount = self.equity * self.risk_pct
                stop_dist = 2 * at