Here's the fully debugged and corrected backtest code with Moon Dev themed improvements:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import numpy as np

class VoltaicSqueeze(Strategy):
    risk_per_trade = 0.02  # 2% risk per trade
    cooldown_period = 3    # bars to wait after 3 consecutive losses
    consecutive_losses = 0
    in_cooldown = 0
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators with TA-Lib using self.I()
        # Bollinger Bands
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name='BB_UPPER', output=['upperband'])
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name='BB_LOWER', output=['lowerband'])
        self.bb_width = self.I(lambda u, l: (u - l), self.bb_upper, self.bb_lower, name='BB_WIDTH')
        
        # Chaikin Volatility
        hl_range = self.I(lambda h, l: h - l, self.data.High, self.data.Low, name='HL_RANGE')
        self.ema_hl = self.I(talib.EMA, hl_range, timeperiod=10, name='EMA_HL')
        
        # Volume MA
        self.volume_ma20 = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VOLUME_MA20')
        
        # ATR for trailing stop
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR14')
        
        print("✨ MOON DEV INIT COMPLETE! All indicators powered up and ready for launch! 🚀")

    def next(self):
        # Cooldown period check
        if self.in_cooldown > 0:
            self.in_cooldown -= 1
            print(f"⏳ MOON DEV COOLDOWN: {self.in_cooldown} bars remaining")
            return

        # Check for consecutive losses
        if self.consecutive_losses >= 3:
            print("🚨 MOON DEV COOLDOWN ACTIVATED! 3 consecutive losses detected ❌")
            self.in_cooldown = self.cooldown_period
            self.consecutive_losses = 0
            return

        # Required minimum data length
        if len(self.data.Close) < 50 or len(self.ema_hl) < 15:
            return

        # Current values
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        bb_width = self.bb_width[-1]
        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]
        volume_ma20 = self.volume_ma20[-1]

        # Bollinger Width percentile (20th percentile of last 50 periods)
        bb_width_history = self.bb_width[-50:]
        width_percentile = np.percentile(bb_width_history, 20)
        bb_compression = bb_width <= width_percentile

        # Chaikin Volatility calculation
        if len(self.ema_hl) >= 11:
            chaikin = ((self.ema_hl[-1] - self.ema_hl[-11]) / self.ema_hl[-11]) * 100
        else:
            chaikin = 0

        # Chaikin Downtrend check (last 5 periods)
        chaikin_trend = self.ema_hl[-5:] if len(self.ema_hl) >=5 else []
        chaikin_downtrend = all(chaikin_trend[i] > chaikin_trend[i+1] for i in range(len(chaikin_trend)-1))

        # Volume spike check
        volume_spike = current_volume >= 2 * volume_ma20

        # Entry conditions
        long_cond = (bb_compression and chaikin_downtrend and 
                    current_close > bb_upper and volume_sp