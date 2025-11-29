Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed indicators:

```python
# 🌙 MOON DEV BACKTESTING SCRIPT - LIQUIDITY BAND REVERSION 🌙

# REQUIRED IMPORTS
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

class LiquidityBandReversion(Strategy):
    # STRATEGY PARAMETERS
    bb_period = 20
    bb_dev = 2
    bandwidth_threshold = 0.5
    rsi_period = 14
    swing_window = 20
    cluster_sensitivity = 0.005  # 0.5%
    risk_pct = 0.01
    max_daily_loss = 0.05
    
    def init(self):
        # 🌙 INDICATOR SETUP USING TA-LIB
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        # Bollinger Bands Components
        self.middle = self.I(talib.SMA, close, timeperiod=self.bb_period)
        self.std = self.I(talib.STDDEV, close, timeperiod=self.bb_period)
        self.upper = self.I(lambda: self.middle + 2*self.std, name='UPPER')
        self.lower = self.I(lambda: self.middle - 2*self.std, name='LOWER')
        self.bandwidth = self.I(lambda: (self.upper - self.lower)/self.middle, name='BANDWIDTH')
        
        # Momentum Indicators
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        
        # Liquidity Zones
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_window, name='SWING_HIGH')
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_window, name='SWING_LOW')
        
        # Volume Validation
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.swing_window)
        
        # Risk Management Tracking
        self.daily_pnl = 0
        self.last_date = None

    def next(self):
        # 🌙 MOON DEV RISK GUARDIAN
        current_date = self.data.index[-1].date()
        if current_date != self.last_date:
            self.daily_pnl = 0
            self.last_date = current_date
            
        if self.daily_pnl <= -self.max_daily_loss * self.equity:
            print(f"🌧️ MOON DEV DAILY LOSS LIMIT TRIGGERED! - ${-self.daily_pnl:.2f}")
            return
            
        # 🌙 CORE STRATEGY LOGIC
        price = self.data.Close[-1]
        upper_zone = self.swing_high[-1]
        lower_zone = self.swing_low[-1]
        
        long_cond = (
            self.bandwidth[-1] < self.bandwidth_threshold and
            price <= lower_zone * (1 + self.cluster_sensitivity) and
            30 < self.rsi[-1] < 70 and
            self.data.Volume[-1] > self.vol_ma[-1]
        )
        
        short_cond = (
            self.bandwidth[-1] < self.bandwidth_threshold and
            price >= upper_zone * (1 - self.cluster_sensitivity) and
            30 < self.rsi[-1] < 70 and
            self.data.Volume[-1] > self.vol_ma[-1]
        )
        
        # 🌙 EXECUTION ENGINE
        if not self.position:
            if long_cond:
                risk_amount = self.equity * self.risk_pct
                sl_price = lower_zone * (1 - 0.01)
                position_size = int(round(risk_amount / (price - sl_price)))
                if position_size > 0:
                    tp_price = price + 1.5*(price - sl_price)
                    self.buy(size=position_size, sl=sl_price, tp=tp_price)
                    print(f"🚀 MOON DEV LONG LAUNCH! | Size: {position_size} | SL: {sl_price:.2f}")
            
            elif short_cond:
                risk_amount = self.equity * self.risk_pct
                sl_price = upper_zone * (1 + 0.01)
                position_size = int(round(risk_amount / (sl_price - price)))
                if position_size > 0:
                    tp_price = price - 1.5*(sl_price - price