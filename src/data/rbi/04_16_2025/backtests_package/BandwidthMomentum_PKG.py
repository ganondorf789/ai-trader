Here's the fixed code with all backtesting.lib imports removed and proper Moon Dev themed indicators:

```python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import talib
from backtesting import Strategy, Backtest

class BandwidthMomentum(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    bb_period = 20
    cmo_period = 14
    sma_period = 50
    lookback_percentile = 100
    volatility_multiplier = 2
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators using talib 🌙
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, self.bb_period, 2, 2, 0, name='BB_UPPER', position=0)
        self.bb_middle = self.I(talib.BBANDS, self.data.Close, self.bb_period, 2, 2, 0, name='BB_MIDDLE', position=1)
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, self.bb_period, 2, 2, 0, name='BB_LOWER', position=2)
        
        self.cmo = self.I(talib.CMO, self.data.Close, self.cmo_period)
        self.cmo_sma = self.I(talib.SMA, self.cmo, self.sma_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
        # Initialize bandwidth tracking
        self.bandwidth_history = []
        self.volatility_avg = []
        
    def next(self):
        # Moon Dev safety check 🌙
        if len(self.data) < max(self.lookback_percentile, self.sma_period + self.cmo_period):
            print("🌙 Not enough data yet - waiting for cosmic alignment...")
            return
            
        # Calculate current bandwidth
        current_bandwidth = (self.bb_upper[-1] - self.bb_lower[-1])/self.bb_middle[-1]*100
        self.bandwidth_history.append(current_bandwidth)
        
        # Volatility contraction check
        if len(self.bandwidth_history) >= self.lookback_percentile:
            perc_10 = np.percentile(self.bandwidth_history[-self.lookback_percentile:], 10)
        else:
            print("🌙 Gathering cosmic bandwidth data...")
            return
            
        # Volatility expansion threshold
        volatility_window = 20
        if len(self.bandwidth_history) >= volatility_window:
            vol_avg = np.mean(self.bandwidth_history[-volatility_window:])
            self.volatility_avg.append(vol_avg)
        else:
            print("🌙 Calculating stellar volatility patterns...")
            return
            
        # Moon Dev Momentum Check 🚀 (replaced crossover with manual check)
        cmo_cross_above = (self.cmo[-2] < self.cmo_sma[-2]) and (self.cmo[-1] > self.cmo_sma[-1])
        cmo_cross_below = (self.cmo[-2] > self.cmo_sma[-2]) and (self.cmo[-1] < self.cmo_sma[-1])
        
        # Entry Logic
        if not self.position:
            # Long Entry 🌕
            if current_bandwidth < perc_10 and cmo_cross_above:
                atr = self.atr[-1]
                stop_loss = self.data.Close[-1] - 1.5*atr
                risk_amount = self.equity * self.risk_percent
                risk_per_share = self.data.Close[-1] - stop_loss
                position_size = int(round(risk_amount/risk_per_share))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_loss)
                    print(f"🌙✨ BULLISH BREAKOUT! Long {position_size} @ {self.data.Close[-1]} | SL: {stop_loss:.2f} 🚀")
                    print(f"🌙 CMO: {self.cmo[-1]:.2f} crossed above SMA: {self.cmo_sma[-1]:.2f}")
            
            # Short Entry 🌑
            elif current_band