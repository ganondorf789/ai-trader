```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

class DynamicNodeBreakout(Strategy):
    # Strategy parameters
    risk_per_trade = 0.01  # 1% risk per trade
    bb_period = 20
    bb_dev = 2
    vwap_lookback = 50
    atr_period = 14
    volume_sma_period = 20
    bb_width_threshold = 0.05
    tp_multiplier = 2
    sl_multiplier = 2
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        self.data.df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']
        
        # Precompute dynamic VWAP
        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        volume = self.data.Volume
        
        dynamic_vwap = []
        anchor_idx = 0
        for i in range(len(self.data)):
            start_idx = max(0, i - self.vwap_lookback + 1)
            window_volume = volume[start_idx:i+1]
            if not window_volume.empty:
                max_vol_idx = window_volume.idxmax()
                if max_vol_idx > anchor_idx:
                    anchor_idx = max_vol_idx
                # Calculate VWAP from anchor to current index
                tp = (high[anchor_idx:i+1] + low[anchor_idx:i+1] + close[anchor_idx:i+1]) / 3
                v = volume[anchor_idx:i+1]
                cum_tpv = (tp * v).cumsum()
                cum_v = v.cumsum()
                current_vwap = cum_tpv[-1] / cum_v[-1] if cum_v[-1] != 0 else np.nan
                dynamic_vwap.append(current_vwap)
            else:
                dynamic_vwap.append(np.nan)
        self.dynamic_vwap = self.I(lambda: pd.Series(dynamic_vwap), name='Dynamic_VWAP')
        
        # Bollinger Bands
        self.bb_upper = self.I(talib.BBANDS, close, self.bb_period, self.bb_dev, self.bb_dev, name='BB_Upper', output=0)
        self.bb_middle = self.I(talib.BBANDS, close, self.bb_period, self.bb_dev, self.bb_dev, name='BB_Middle', output=1)
        self.bb_lower = self.I(talib.BBANDS, close, self.bb_period, self.bb_dev, self.bb_dev, name='BB_Lower', output=2)
        
        # ATR for stop loss and take profit
        self.atr = self.I(talib.ATR, high, low, close, self.atr_period, name='ATR')
        
        # Volume SMA
        self.volume_sma = self.I(talib.SMA, volume, self.volume_sma_period, name='Volume_SMA')
        
        print("🌙 Moon Dev Indicators Activated! Dynamic VWAP Anchored 🚀")

    def next(self):
        # Skip calculation for initial bars
        if len(self.data) < self.vwap_lookback:
            return
        
        current_close = self.data.Close[-1]
        dynamic_vwap = self.dynamic_vwap[-1]
        prev_close = self.data.Close[-2]
        prev_vwap = self.dynamic_vwap[-2]
        
        # Calculate Bollinger Band Width
        bb_width = (self.bb_upper[-1] - self.bb_lower[-1]) / self.bb_middle[-1]
        
        # Volume check
        current_volume = self.data.Volume[-1]
        volume_sma = self.volume_sma[-1]
        
        # Entry conditions
        long_cond = (current_close > dynamic_vwap and 
                    prev_close <= prev_vwap and 
                    bb_width > self.bb_width_threshold and 
                    current_volume > volume_sma)
        
        short_cond = (current_close < dynamic_vwap and 
                     prev_close >= prev_vwap and 
                     bb_width > self.bb_width_threshold and 
                     current_volume > volume_sma)
        
        # Risk management calculations
        equity = self.equ