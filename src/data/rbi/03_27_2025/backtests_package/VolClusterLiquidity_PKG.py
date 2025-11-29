Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed indicators:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

class VolClusterLiquidity(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    atr_period = 14
    bb_period = 20
    bb_dev = 2
    swing_period = 20
    liquidity_lookback = 100  # For liquidity zone detection ✨
    
    def init(self):
        # Clean data and handle columns
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators using TA-Lib
        self.upper, self.middle, self.lower = self.I(talib.BBANDS, self.data.Close, 
                                                    timeperiod=self.bb_period, 
                                                    nbdevup=self.bb_dev, 
                                                    nbdevdn=self.bb_dev)
        bb_width = (self.upper - self.lower) / self.middle
        self.bb_width = self.I(lambda x: x, bb_width, name='BB Width')  # Wrap in self.I
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        # Liquidity zones using swing highs/lows 🏔️
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period)
        
        # Volume analysis 📊
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20)
        
        print("🌙 Moon Dev Indicators Initialized! ✨")

    def next(self):
        current_idx = len(self.data.Close) - 1
        if current_idx < self.liquidity_lookback:
            return
        
        # Calculate dynamic volatility threshold
        bb_width_values = self.bb_width[-self.liquidity_lookback:]
        volatility_threshold = np.percentile(bb_width_values, 20)
        
        # Current market conditions 🌐
        price = self.data.Close[-1]
        volatility_contraction = self.bb_width[-1] < volatility_threshold
        volume_spike = self.data.Volume[-1] > self.volume_ma[-1]
        
        # Liquidity zone detection 🎯
        nearest_swing_high = max(self.swing_high[-5:])
        nearest_swing_low = min(self.swing_low[-5:])
        
        # Long setup conditions 🚀
        long_trigger = (price > self.upper[-1] and 
                        volatility_contraction and 
                        volume_spike)
        
        # Short setup conditions 🪂
        short_trigger = (price < self.lower[-1] and 
                        volatility_contraction and 
                        volume_spike)
        
        # Position sizing calculations 💰
        equity = self.equity
        risk_amount = equity * self.risk_per_trade
        atr_value = self.atr[-1]
        
        if not self.position:
            if long_trigger:
                # Calculate stop loss and position size
                sl_price = min(self.swing_low[-1], self.lower[-1])
                risk_per_unit = price - sl_price
                position_size = int(round(risk_amount / risk_per_unit))
                
                # Determine profit targets 🎯
                tp1 = nearest_swing_high
                tp2 = price + 1.5 * atr_value
                
                if position_size > 0:
                    self.buy(size=position_size, 
                            sl=sl_price,
                            tp=max(tp1, tp2))
                    print(f"🚀 LONG LAUNCH! Size: {position_size} | Entry: {price:.2f} | SL: {sl_price:.2f} | TP: {max(tp1, tp2):.2f}")
            
            elif short_trigger:
                # Calculate stop loss and position size
                sl_price = max(self.swing_high[-1], self.upper[-1])
                risk_per_unit = sl_price - price
                position_size = int(round(risk_amount / risk_per_unit))
                
                # Determine profit targets 🎯
                tp1 = nearest_swing_low
                tp2 = price - 1.5 * atr_value