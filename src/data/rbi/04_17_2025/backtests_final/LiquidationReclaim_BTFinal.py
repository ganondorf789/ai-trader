Here's the complete fixed backtest code with Moon Dev themed improvements, ensuring all technical requirements are met while preserving the original strategy logic:

```python
# 🌙✨ Moon Dev Backtest AI - Liquidation Reclaim Strategy 🚀
# 🌙 CRITICAL: Position Sizing Rules Verified - Whole Numbers for Units ✅

import pandas as pd
import talib
from backtesting import Strategy, Backtest

class LiquidationReclaim(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade (fractional sizing)
    atr_period = 14
    
    def init(self):
        # 🌙✨ Indicator Initialization Phase
        # Volume Analysis
        self.hourly_volume = self.I(talib.SUM, self.data.Volume, 4, name='Hourly_Vol')
        self.hourly_mean = self.I(talib.SMA, self.hourly_volume, 80, name='Hourly_Mean')
        self.hourly_std = self.I(talib.STDDEV, self.hourly_volume, 80, name='Hourly_Std')
        
        # VWAP Calculation
        def calc_vwap(h, l, c, v):
            typical = (h + l + c) / 3
            cumulative_tpv = typical * v
            cumulative_vol = v.cumsum()
            return cumulative_tpv.cumsum() / cumulative_vol
        self.vwap = self.I(calc_vwap, self.data.High, self.data.Low, self.data.Close, 
                          self.data.Volume, name='VWAP')
        
        # Bollinger Bands
        self.upper_bb, self.middle_bb, self.lower_bb = self.I(
            lambda close: talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2),
            self.data.Close,
            name=['Upper_BB', 'Middle_BB', 'Lower_BB']
        )
        
        # Risk Management Indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                          self.atr_period, name='ATR')
        self.swing_low = self.I(talib.MIN, self.data.Low, 20, name='Swing_Low')
        
        print("🌙✨ Moon Dev Indicators Initialized - Ready for Cosmic Trading! 🚀")

    def next(self):
        # 🌙✨ Trading Logic Execution
        current_close = self.data.Close[-1]
        
        # Calculate Z-Score for liquidation detection
        current_vol = self.hourly_volume[-1]
        mean_vol = self.hourly_mean[-1]
        std_vol = self.hourly_std[-1]
        z_score = (current_vol - mean_vol) / std_vol if std_vol != 0 else 0
        
        # Entry Conditions
        liquidation_cond = z_score > 2
        vwap_cond = current_close > self.vwap[-1]
        volume_cond = (self.data.Volume[-1] > self.data.Volume[-2] and 
                      self.data.Volume[-2] > self.data.Volume[-3])
        
        # 🌙 Position Management
        if not self.position:
            # Long Entry Logic
            if liquidation_cond and vwap_cond and volume_cond:
                stop_loss = min(self.swing_low[-1], self.vwap[-1])
                risk_per_share = current_close - stop_loss
                
                if risk_per_share > 0:
                    # 🌙 CRITICAL FIX: Proper position sizing with whole numbers
                    risk_amount = self.risk_per_trade * self.equity
                    position_size = int(round(risk_amount / risk_per_share))
                    
                    if position_size > 0:  # Ensure valid position size
                        self.buy(size=position_size, sl=stop_loss)
                        print(f"🌙 BUY Signal: {position_size} units at {current_close}")
        
        else:  # Existing position management
            # Bollinger Squeeze Exit Logic
            bb_width = (self.upper_bb[-1] - self.lower_bb[-1]) / self.middle_bb[-1]
            squeeze_cond = bb_width < 0.1  # Threshold for squeeze
            
            if squeeze_cond and current_close > self.upper_bb[-1]:
                self.position.close()
                print("🌙 EXIT Signal: Bollinger Squeeze Breakout")
            
            # ATR Trailing Stop
            if current_close > self.position.entry_price + self.atr[-1]:
                new_sl = current_close - 2