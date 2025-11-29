Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations:

```python
# 🌙 Moon Dev's VolterraSqueeze Backtest Implementation ✨
# 🚀 Package AI Verified - No backtesting.lib imports! 🌌

import pandas as pd
import talib
from backtesting import Backtest, Strategy

class VolterraSqueeze(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌑
    bb_period = 20
    kc_period = 20
    atr_period = 14

    def init(self):
        # 🌙✨ Indicator Calculations using TA-Lib
        # Bollinger Bands
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, 
                              timeperiod=self.bb_period, nbdevup=2, 
                              name='BB_Upper')
        self.bb_mid = self.I(talib.BBANDS, self.data.Close,
                            timeperiod=self.bb_period, nbdevup=2,
                            name='BB_Mid', which=1)
        self.bb_lower = self.I(talib.BBANDS, self.data.Close,
                              timeperiod=self.bb_period, nbdevup=2,
                              name='BB_Lower', which=2)
        
        # Keltner Channel 
        self.kc_mid = self.I(talib.EMA, self.data.Close, 
                            timeperiod=self.kc_period, name='KC_Mid')
        self.kc_atr = self.I(talib.ATR, self.data.High, self.data.Low,
                            self.data.Close, timeperiod=self.kc_period)
        self.kc_upper = self.I(lambda mid, atr: mid + 2*atr,
                              self.kc_mid, self.kc_atr, name='KC_Upper')
        self.kc_lower = self.I(lambda mid, atr: mid - 2*atr,
                              self.kc_mid, self.kc_atr, name='KC_Lower')
        
        # Volatility Metrics
        self.bb_width = self.I(lambda u,l,m: (u-l)/m,
                             self.bb_upper, self.bb_lower, self.bb_mid,
                             name='BB_Width')
        self.bb_pct = self.I(lambda x: x.rolling(20).quantile(0.2),
                            self.bb_width, name='BB_Pct')
        
        # Volume and ATR
        self.vol_sma = self.I(talib.SMA, self.data.Volume, 20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low,
                         self.data.Close, self.atr_period)
        
        # Track moon phases 🌓🌔🌕
        self.consecutive_losses = 0

    def next(self):
        # 🌑 Check cosmic alignment (3 consecutive losses)
        if len(self.trades) >= 3:
            last_three = [t for t in self.trades[-3:] if t.pl <= 0]
            if len(last_three) >= 3:
                print("🚨 Cosmic Warning: 3 Losses! Trading Halted 🌌")
                return

        if not self.position:
            # 🌙✨ Entry Logic
            close = self.data.Close[-1]
            vol_ratio = self.data.Volume[-1] / self.vol_sma[-1]
            
            # Long Conditions 🌕
            long_cond = (
                self.bb_pct[-1] <= 0.2 and
                self.kc_upper[-1] < self.kc_upper[-2] and
                close > self.bb_upper[-1] and
                close > self.kc_upper[-1] and
                vol_ratio >= 1.5
            )
            
            # Short Conditions 🌑
            short_cond = (
                self.bb_pct[-1] <= 0.2 and
                self.kc_lower[-1] > self.kc_lower[-2] and
                close < self.bb_lower[-1] and
                close < self.kc_lower[-1] and
                vol_ratio >= 1.5
            )
            
            # Calculate position size 🌙
            atr_val = self.atr[-1]
            risk_amount = self.risk_percent * self.equity
            
            if long_cond:
                sl = max(close - 0.75*atr_val, 
                        self.I(talib.MIN, self.data.Low, 20)[-1])
                size = int(risk_amount / (