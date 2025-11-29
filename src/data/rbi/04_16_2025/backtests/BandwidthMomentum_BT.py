```python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import talib
from backtesting import Strategy, Backtest
from backtesting.lib import crossover

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
        
        # Calculate indicators
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
            return
            
        # Calculate current bandwidth
        current_bandwidth = (self.bb_upper[-1] - self.bb_lower[-1])/self.bb_middle[-1]*100
        self.bandwidth_history.append(current_bandwidth)
        
        # Volatility contraction check
        if len(self.bandwidth_history) >= self.lookback_percentile:
            perc_10 = np.percentile(self.bandwidth_history[-self.lookback_percentile:], 10)
        else:
            return
            
        # Volatility expansion threshold
        volatility_window = 20
        if len(self.bandwidth_history) >= volatility_window:
            vol_avg = np.mean(self.bandwidth_history[-volatility_window:])
            self.volatility_avg.append(vol_avg)
        else:
            return
            
        # Moon Dev Momentum Check 🚀
        cmo_cross_above = crossover(self.cmo, self.cmo_sma)
        cmo_cross_below = crossover(-self.cmo, -self.cmo_sma)
        
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
            
            # Short Entry 🌑
            elif current_bandwidth < perc_10 and cmo_cross_below:
                atr = self.atr[-1]
                stop_loss = self.data.Close[-1] + 1.5*atr
                risk_amount = self.equity * self.risk_percent
                risk_per_share = stop_loss - self.data.Close[-1]
                position_size = int(round(risk_amount/risk_per_share))
                
                if position_size > 0:
                    self.sell(size=position_size, sl=stop_loss)
                    print(f"🌙✨ BEARISH BREAKOUT! Short {position_size} @ {self.data.Close[-1]} | SL