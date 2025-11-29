```python
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
from backtesting import Strategy, Backtest

class LiquidationClusterReversal(Strategy):
    risk_per_trade = 0.02  # 2% risk per trade
    max_positions = 2
    cooldown_hours = 24
    atr_period = 14
    atr_ma_period = 20
    liquidation_std_period = 30
    cluster_lookback = 5
    cluster_threshold = 3
    
    def init(self):
        # Clean data columns
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        
        # Calculate indicators
        self.liquidation_ma = self.I(talib.SMA, self.data.df['liquidation'], self.liquidation_std_period, name='LIQ_MA')
        self.liquidation_std = self.I(talib.STDDEV, self.data.df['liquidation'], self.liquidation_std_period, name='LIQ_STD')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR')
        self.atr_ma = self.I(talib.SMA, self.atr, self.atr_ma_period, name='ATR_MA')
        self.vwap = self.I(ta.vwap, high=self.data.High, low=self.data.Low, close=self.data.Close, 
                          volume=self.data.Volume, name='VWAP')
        
        # Track trade history
        self.last_long_entry = None
        self.last_short_entry = None
        self.position_size = None

    def next(self):
        current_close = self.data.Close[-1]
        current_time = self.data.index[-1]
        
        # Liquidation bands calculations
        upper_band = self.liquidation_ma[-1] + 2*self.liquidation_std[-1]
        lower_band = self.liquidation_ma[-1] - 2*self.liquidation_std[-1]
        current_liq = self.data.df['liquidation'][-1]
        
        # Cluster detection
        long_cluster = short_cluster = False
        liq_vals = self.data.df['liquidation'][-self.cluster_lookback:]
        ma_vals = self.liquidation_ma[-self.cluster_lookback:]
        std_vals = self.liquidation_std[-self.cluster_lookback:]
        
        long_cluster_count = sum(1 for i in range(len(liq_vals)) 
                               if liq_vals.iloc[i] < (ma_vals.iloc[i] - 2*std_vals.iloc[i]))
        short_cluster_count = sum(1 for i in range(len(liq_vals)) 
                                if liq_vals.iloc[i] > (ma_vals.iloc[i] + 2*std_vals.iloc[i]))
        
        # Entry conditions
        long_cond = (current_liq < lower_band and
                    long_cluster_count >= self.cluster_threshold and
                    abs(current_close - lower_band) < 0.5*self.atr[-1] and
                    self.atr[-1] < self.atr_ma[-1])
        
        short_cond = (current_liq > upper_band and
                     short_cluster_count >= self.cluster_threshold and
                     abs(current_close - upper_band) < 0.5*self.atr[-1] and
                     self.atr[-1] < self.atr_ma[-1])
        
        # Risk management checks
        valid_long = (self.last_long_entry is None or 
                     (current_time - self.last_long_entry).total_seconds()/3600 >= self.cooldown_hours)
        valid_short = (self.last_short_entry is None or 
                      (current_time - self.last_short_entry).total_seconds()/3600 >= self.cooldown_hours)
        
        # Execute entries
        if not self.position and len(self.trades) < self.max_positions:
            if long_cond and valid_long:
                self.enter_long(current_close, current_time)
            elif short_cond and valid_short:
                self.enter_short(current_close, current_time)
                
        # Exit conditions
        for trade in self.trades:
            if trade.is_long:
                if self.atr[-1] > self.atr_ma[-1]*1.15 or current_close < self.vwap[-1]:
                    trade