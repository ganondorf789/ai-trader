```python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class VoltaicSqueeze(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    atr_multiplier = 2
    bb_percentile_thresh = 20
    emergency_exit_thresh = 80
    
    def init(self):
        # Calculate indicators using TA-Lib/pandas_ta with self.I()
        self.bb_upper = self.I(self._calc_bb_upper)
        self.bb_lower = self.I(self._calc_bb_lower)
        self.bb_percentile = self.I(self._calc_bb_percentile)
        self.kc_upper = self.I(self._calc_kc_upper)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.atr_ma = self.I(talib.SMA, self.atr, 20)
        
        # Track highest close for trailing stop
        self.highest_close = None
        
    def _calc_bb_upper(self):
        mid = talib.SMA(self.data.Close, 20)
        std = talib.STDDEV(self.data.Close, 20)
        return mid + 2 * std
    
    def _calc_bb_lower(self):
        mid = talib.SMA(self.data.Close, 20)
        std = talib.STDDEV(self.data.Close, 20)
        return mid - 2 * std
    
    def _calc_bb_percentile(self):
        width = self.bb_upper - self.bb_lower
        max_width = talib.MAX(width, 100)
        percentile = (width / np.maximum(max_width, 1e-6)) * 100  # Avoid division by zero
        return percentile
    
    def _calc_kc_upper(self):
        kc = ta.kc(
            high=self.data.High, 
            low=self.data.Low, 
            close=self.data.Close,
            length=20,
            scalar=1.5
        )
        return kc['KCU_20_1.5']
    
    def next(self):
        if not self.position:
            self._check_entries()
        else:
            self._manage_exits()
    
    def _check_entries(self):
        # Entry conditions
        cond_vol_contraction = self.bb_percentile[-1] < self.bb_percentile_thresh
        cond_breakout = self.data.Close[-1] > self.kc_upper[-1]
        cond_vol_filter = self.atr[-1] > self.atr_ma[-1]
        
        if cond_vol_contraction and cond_breakout and cond_vol_filter:
            entry_price = self.data.Open[-1]  # Next candle's open
            atr_value = self.atr[-1]
            stop_loss = entry_price - self.atr_multiplier * atr_value
            risk_per_share = entry_price - stop_loss
            
            if risk_per_share <= 0:
                print("🌙✨ Risk per share <=0, skipping entry")
                return
            
            position_size = self._calc_position_size(risk_per_share)
            if position_size > 0:
                self.buy(size=position_size, tag="VoltaicSqueeze Entry")
                self.highest_close = entry_price
                print(f"🚀🌕 MOON DEV ENTRY: Long {position_size} @ {entry_price:.2f}")
    
    def _calc_position_size(self, risk_per_share):
        risk_amount = self.equity * self.risk_percent
        size = risk_amount / risk_per_share
        return int(round(size))
    
    def _manage_exits(self):
        if self.position.is_long:
            self.highest_close = max(self.highest_close, self.data.Close[-1])
            current_stop = self.highest_close - self.atr_multiplier * self.atr[-1]
            
            # Trailing stop exit
            if self.data.Low[-1] < current_stop:
                self.position.close()
                print(f"🌑🛑 MOON DEV TRAILING STOP: {self.data.Close[-1]:.2f}")
            
            # Emergency volatility exit
            elif self.bb_percentile[-1] > self.emergency_exit_thresh:
                self.position.close()
                print(f"🌪️🔥 MOON DEV VOLATILITY EXIT: {self.data.Close[-1]:.2f