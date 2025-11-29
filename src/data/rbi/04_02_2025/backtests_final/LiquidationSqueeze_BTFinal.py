I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete debugged version with Moon Dev improvements:

```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR LIQUIDATION SQUEEZE STRATEGY 🚀✨
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

class LiquidationSqueeze(Strategy):
    risk_per_trade = 0.01  # 1% of equity per trade
    
    def init(self):
        # 🌙 INDICATOR INITIALIZATION PHASE
        # Liquidation Cluster Detection (Swing Highs/Lows)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='🌙 SWING HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='✨ SWING LOW')
        
        # Bollinger Bands with Width Calculation
        def bb_upper(close):
            upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return upper
        self.bb_upper = self.I(bb_upper, self.data.Close, name='🌙 BB UPPER')
        
        self.bb_lower = self.I(lambda close: talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)[2], 
                              self.data.Close, name='✨ BB LOWER')
        self.bb_width = self.I(lambda u,l: u-l, self.bb_upper, self.bb_lower, name='🌙 BB WIDTH')
        self.bb_width_avg = self.I(talib.SMA, self.bb_width, timeperiod=50, name='✨ BB WIDTH AVG')
        
        # Volatility and Volume Indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='🌙 ATR 14')
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='✨ VOLUME MA 20')
        
        print("🌙 MOON DEV INDICATORS READY FOR LIFTOFF! 🚀")
        print("✨ ALL SYSTEMS GO - NO BACKTESTING.LIB DETECTED 🌙")

    def next(self):
        # 🌙 STRATEGY LOGIC EXECUTION
        price = self.data.Close[-1]
        swing_high = self.swing_high[-1]
        swing_low = self.swing_low[-1]
        
        # 💎 BOLLINGER SQUEEZE CONDITION
        squeeze_condition = (self.bb_width[-1] < 0.5 * self.bb_width_avg[-1])
        
        # 🚀 VOLUME CONFIRMATION
        volume_spike = self.data.Volume[-1] > self.volume_ma[-1]
        
        if not self.position:
            # 🌙 LONG ENTRY LOGIC
            if squeeze_condition and volume_spike and price > swing_high:
                midpoint = (swing_high + swing_low)/2
                risk_amount = self.risk_per_trade * self.equity
                risk_per_share = price - midpoint
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    tp_level = swing_high + 1.5 * self.atr[-1]
                    
                    if position_size > 0:
                        self.buy(size=position_size, 
                                sl=midpoint,
                                tp=tp_level,
                                tag="🌙 MOON SHOT LONG 🚀")
                        print(f"🌙 LIFTOFF! LONG {position_size} units @ {price:.2f}")
                        print(f"   🛡️ SL: {midpoint:.2f} | 🎯 TP: {tp_level:.2f}")
            
            # ✨ SHORT ENTRY LOGIC
            elif squeeze_condition and volume_spike and price < swing_low:
                midpoint = (swing_high + swing_low)/2
                risk_amount = self.risk_per_trade * self.equity
                risk_per_share = midpoint - price
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    tp_level = swing_low - 1.5 * self.atr[-1]
                    
                    if position_size > 0:
                        self.sell(size=position_size,
                                 sl=midpoint,