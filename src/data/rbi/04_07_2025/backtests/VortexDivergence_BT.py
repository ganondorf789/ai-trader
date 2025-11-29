```python
# 🌙 Moon Dev's VortexDivergence Backtest 🌙
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta

class VortexDivergence(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    atr_multiplier = 2  # 2x ATR for trailing stop 🚀
    vi_period = 14
    swing_period = 5  # Swing high/low lookback
    
    def init(self):
        # 🌙 Clean data columns handled externally
        
        # Calculate indicators with self.I()
        self.vi_plus, self.vi_minus = self.I(self.calculate_vortex)
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
        # Swing high/low indicators
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period)
        self.obv_high = self.I(talib.MAX, self.obv, self.swing_period)
        self.obv_low = self.I(talib.MIN, self.obv, self.swing_period)
        
        self.trailing_stop = None  # Dynamic trailing stop 🌊

    def calculate_vortex(self):
        # TA-LIB Vortex implementation 🌪️
        high = self.data.High.astype(float).values
        low = self.data.Low.astype(float).values
        close = self.data.Close.astype(float).values
        
        # Calculate VM+ and VM-
        vm_plus = np.abs(high[1:] - low[:-1])
        vm_minus = np.abs(low[1:] - high[:-1])
        
        # Pad with NaN to maintain array length
        vm_plus = np.insert(vm_plus, 0, np.nan)
        vm_minus = np.insert(vm_minus, 0, np.nan)
        
        # True Range calculation
        tr = talib.TRANGE(high, low, close)
        
        # Sum components
        sum_vm_plus = talib.SUM(vm_plus, self.vi_period)
        sum_vm_minus = talib.SUM(vm_minus, self.vi_period)
        sum_tr = talib.SUM(tr, self.vi_period)
        
        # Avoid division by zero
        np.seterr(divide='ignore', invalid='ignore')
        vi_plus = np.divide(sum_vm_plus, sum_tr)
        vi_minus = np.divide(sum_vm_minus, sum_tr)
        
        return vi_plus, vi_minus

    def next(self):
        if len(self.data) < self.swing_period + 2:
            return  # Wait for enough data 🌙
            
        current_close = self.data.Close[-1]
        current_atr = self.atr[-1]
        
        # 🌙 Avoid low volatility periods
        if current_atr < (np.nanmean(self.atr)*0.5):
            return  # Skip trades when ATR < 50% of average ATR 🌊
            
        # ===== ENTRY LOGIC =====
        if not self.position:
            # Bullish divergence check 🌟
            price_higher_low = self.data.Low[-1] > self.swing_low[-2]
            obv_higher_high = self.obv[-1] > self.obv_high[-2]
            
            # Bearish divergence check 🌑
            price_lower_high = self.data.High[-1] < self.swing_high[-2]
            obv_lower_low = self.obv[-1] < self.obv_low[-2]
            
            # Entry conditions 🚀
            if crossover(self.vi_plus, self.vi_minus) and price_higher_low and obv_higher_high:
                # Calculate position size 🌙
                risk_amount = self.equity * self.risk_per_trade
                stop_loss = self.swing_low[-1]
                risk_per_share = current_close - stop_loss
                
                if risk_per_share <= 0:
                    return  # Invalid SL
                    
                position_size = int(round(risk_amount / risk_per_share))