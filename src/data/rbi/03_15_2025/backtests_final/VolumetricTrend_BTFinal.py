Here's the complete fixed code with Moon Dev themed debug prints and all necessary corrections applied. I've focused on fixing technical issues while preserving the original strategy logic. 🌙✨

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Data preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolumetricTrend(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    atr_multiple = 1.5
    ma_short = 10
    ma_long = 50
    volume_window = 20
    atr_window = 14

    def init(self):
        # Initialize indicators using talib
        self.sma10 = self.I(talib.SMA, self.data.Close, timeperiod=self.ma_short)
        self.sma50 = self.I(talib.SMA, self.data.Close, timeperiod=self.ma_long)
        self.vol_avg = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_window)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_window)

    def next(self):
        # Moon Dev Safety Checks 🌙
        if len(self.sma10) < 2 or len(self.sma50) < 2:
            print("🌙✨ Not enough data for crossover detection. Skipping...")
            return

        # Current indicator values
        sma10_now = self.sma10[-1]
        sma10_prev = self.sma10[-2]
        sma50_now = self.sma50[-1]
        sma50_prev = self.sma50[-2]
        volume_now = self.data.Volume[-1]
        vol_avg_now = self.vol_avg[-1]
        atr_now = self.atr[-1]

        # Entry Logic 🚀
        if not self.position:
            # Long entry (bullish crossover)
            if (sma10_prev <= sma50_prev) and (sma10_now > sma50_now) and (volume_now > vol_avg_now):
                risk_amount = self.equity * self.risk_percent
                risk_per_unit = self.atr_multiple * atr_now
                
                if risk_per_unit <= 0:
                    print("🌙✨ ATR Value Too Low - Trade Aborted!")
                    return
                
                position_size = int(round(risk_amount / risk_per_unit))
                if position_size > 0:
                    self.buy(size=position_size)
                    print(f"🌙✨ BULLISH CROSSOVER DETECTED! Entered Long at {self.data.Close[-1]} 🚀")
                    print(f"    Size: {position_size} BTC | Risk: {self.risk_percent*100}% | ATR Stop: {risk_per_unit:.2f}")

            # Short entry (bearish crossover)
            elif (sma10_prev >= sma50_prev) and (sma10_now < sma50_now) and (volume_now > vol_avg_now):
                risk_amount = self.equity * self.risk_percent
                risk_per_unit = self.atr_multiple * atr_now
                
                if risk_per_unit <= 0:
                    print("🌙✨ ATR Value Too Low - Trade Aborted!")
                    return
                
                position_size = int(round(risk_amount / risk_per_unit))
                if position_size > 0:
                    self.sell(size=position_size)
                    print(f"🌙✨ BEARISH CROSSOVER DETECTED! Entered Short at {self.data.Close[-1]} 📉")
                    print(f"    Size: {position_size} BTC | Risk: {self.risk_percent*100}% | ATR Stop: {risk_per_unit:.2f}")

        # Exit Logic 🛑
        else:
            if not hasattr(self, 'entry_atr'):
                self.entry_atr = self.atr[-1]
                self.entry