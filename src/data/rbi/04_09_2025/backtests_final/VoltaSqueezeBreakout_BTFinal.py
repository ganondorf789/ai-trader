I'll debug the VoltaSqueezeBreakout strategy while maintaining all the original logic. Here's the complete fixed code with Moon Dev themed debug prints:

```python
# 🌙 Moon Dev's VoltaSqueezeBreakout Backtest 🌙
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np
from backtesting import Backtest, Strategy

class VoltaSqueezeBreakout(Strategy):
    # Strategy parameters
    risk_per_trade = 0.01  # 1% risk per trade
    max_daily_risk = 0.05  # 5% max daily loss
    
    def init(self):
        # 🌙 Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col], inplace=True)
        
        # 🌟 Core Indicators Calculation
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume
        
        # 🎯 Bollinger Bands (20,2)
        def bb_calculation(close):
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            return upper, middle, lower
        
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(bb_calculation, close, name=['BB_Upper', 'BB_Middle', 'BB_Lower'])
        
        # 📈 Bollinger Bandwidth (20,2)
        self.bb_bandwidth = self.I(lambda u, l, m: (u - l)/m, 
                                  self.bb_upper, self.bb_lower, self.bb_middle,
                                  name='BB_Bandwidth')
        
        # 🎯 Keltner Channel (20,1.5)
        self.ema_kc = self.I(talib.EMA, close, timeperiod=20, name='EMA_KC')
        self.atr_kc = self.I(talib.ATR, high, low, close, timeperiod=20, name='ATR_KC')
        self.upper_kc = self.I(lambda e, a: e + 1.5*a, self.ema_kc, self.atr_kc, name='Upper_KC')
        self.lower_kc = self.I(lambda e, a: e - 1.5*a, self.ema_kc, self.atr_kc, name='Lower_KC')
        
        # 🌊 Volume Conditions
        self.volume_sma20 = self.I(talib.SMA, volume, timeperiod=20, name='Volume_SMA20')
        
        # 📉 Trend Detection
        self.kc_trend = self.I(talib.LINEARREG_SLOPE, self.ema_kc, timeperiod=5, name='KC_Trend')
        
        # 🎯 Risk Management Indicators
        self.atr14 = self.I(talib.ATR, high, low, close, timeperiod=14, name='ATR14')
        
        # 📊 Bandwidth Percentile Calculation
        self.bb_percentile = self.I(ta.percentile, self.bb_bandwidth, length=200, q=10, name='BB_Percentile')
        
        # 🌙 Track daily performance
        self.last_day = None
        self.daily_pnl = 0
        self.trade_duration = 0

    def next(self):
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        
        # 🌙 Check daily loss limit
        today = self.data.index[-1].date()
        if today != self.last_day:
            self.last_day = today
            self.daily_pnl = 0
            print(f"🌙✨ New trading day: {today}")
        
        # 🛑 Prevent trading if daily loss exceeded
        if self.daily_pnl <= -self.max_daily_risk * self.equity:
            print(f"🌙✨🛑 Daily loss limit hit! No new trades today.")
            return
            
        if not self.position:
            # 🌟 Long Entry Conditions
            long_cond = (
                self.bb_bandwidth[-1] < self.bb_percentile[-1] and
                current_close > self.bb_upper[-1] and
                current_volume > 2 * self.volume_sma20[-1] and
                self.kc_trend[-1] > 0
            )