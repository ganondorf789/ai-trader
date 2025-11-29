Here's the fixed code with proper indicator packages and Moon Dev themed debug prints, ensuring no backtesting.lib imports are used:

```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

class HeikinTickDivergence(Strategy):
    initial_equity = 1_000_000
    risk_per_trade = 0.01
    atr_period = 14
    daily_loss_limit = 0.05
    
    def init(self):
        # Clean and prepare Heikin-Ashi data 🌙✨
        ha_close = (self.data.Open + self.data.High + self.data.Low + self.data.Close) / 4
        ha_open = pd.Series(0.0, index=self.data.index)
        ha_open[0] = (self.data.Open[0] + self.data.Close[0]) / 2
        
        for i in range(1, len(self.data)):
            ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
            
        ha_high = pd.concat([self.data.High, ha_open, ha_close], axis=1).max(axis=1)
        ha_low = pd.concat([self.data.Low, ha_open, ha_close], axis=1).min(axis=1)

        # Moon Dev's Indicator Suite 🌙📊
        self.ha_open = self.I(lambda: ha_open, name='HA_Open')
        self.ha_close = self.I(lambda: ha_close, name='HA_Close')
        self.ha_high = self.I(lambda: ha_high, name='HA_High')
        self.ha_low = self.I(lambda: ha_low, name='HA_Low')
        
        # Technical indicators using proper TA-Lib calls 🌙🔍
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.price_slope = self.I(talib.LINEARREG_SLOPE, self.data.Close, timeperiod=3)
        self.volume_slope = self.I(talib.LINEARREG_SLOPE, self.data.Volume, timeperiod=3)
        
        # Session tracking 🌙⏳
        self.current_day = None
        self.trades_today = 0

    def next(self):
        # Moon Dev's Cosmic Timekeeper 🌙⏰
        current_day = self.data.index[-1].date()
        if current_day != self.current_day:
            self.current_day = current_day
            self.trades_today = 0
            if self.equity < self.initial_equity * (1 - self.daily_loss_limit):
                print(f"🌙🚨 MOON DEV ALERT: Daily loss limit triggered! No trades for {current_day}")
                return

        # Avoid first/last 30 minutes (BTC 24h market)
        current_minute = self.data.index[-1].minute
        current_hour = self.data.index[-1].hour
        if (current_hour == 0 and current_minute < 30) or (current_hour == 23 and current_minute >= 30):
            return

        # Core strategy logic 🌙🚀
        if not self.position:
            if len(self.ha_close) < 3:
                return

            # Heikin-Ashi trend confirmation using pure numpy/pandas
            ha_bullish = all(self.ha_close[-i] > self.ha_open[-i] for i in range(1,4))
            ha_bearish = all(self.ha_close[-i] < self.ha_open[-i] for i in range(1,4))
            
            # Volume divergence detection
            price_dir = self.price_slope[-1]
            vol_dir = self.volume_slope[-1]
            
            if ha_bullish and price_dir <= 0 and vol_dir > 0:
                entry_level = max(self.ha_high[-3:])
                if self.data.Close[-1] > entry_level:
                    self.enter_long()
                    print("🌙🚀 LONG position entered - Bullish HA with volume divergence!")
                    
            elif ha_bearish and price_dir >= 0 and vol_dir < 0:
                entry_level = min(self.ha_low[-3:])
                if self.data.Close[-1] < entry_level:
                    self.enter_short()
                    print("🌙🌌 SHORT position entered - Bearish HA with volume divergence!")
        else:
            # Exit conditions 🌙🛑
            if self.position.is_long:
                if self.ha_close[-1] < self.ha_open