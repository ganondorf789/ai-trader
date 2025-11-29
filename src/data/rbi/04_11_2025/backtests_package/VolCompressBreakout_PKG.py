Here's the fixed code with all backtesting.lib imports and functions removed, replaced with proper implementations, and Moon Dev themed debug prints:

```python
# 🌙 Moon Dev Backtest Implementation for VolCompress Breakout Strategy
import pandas as pd
import talib
import numpy as np
from backtesting import Strategy, Backtest

class VolCompressBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌙 Indicator Calculation Phase
        # Bollinger Bands components
        def bb_upper(close):
            upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            return upper
        self.bb_upper = self.I(bb_upper, self.data.Close, name='BB_UPPER')
        
        def bb_lower(close):
            _, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            return lower
        self.bb_lower = self.I(bb_lower, self.data.Close, name='BB_LOWER')
        
        # Bollinger Band Width (BBW)
        self.bbw = self.I(lambda u, l: (u - l) / ((u + l)/2), 
                          self.bb_upper, self.bb_lower, name='BBW')
        
        # BBW 20th percentile (100-period lookback)
        self.bbw_percentile = self.I(
            lambda x: x.rolling(100).quantile(0.2), self.bbw, name='BBW_PERCENTILE')
        
        # Volume indicators
        self.volume_avg = self.I(talib.SMA, self.data.Volume, 20, name='VOL_MA20')
        
        # Ichimoku Kijun-sen (Base Line)
        def get_kijun(high, low):
            _, kijun, _, _, _ = talib.ICHIMOKU(high, low, 9, 26, 52)
            return kijun
        self.kijun = self.I(get_kijun, self.data.High, self.data.Low, name='KIJUN')
        
        # Swing High/Low
        self.swing_high = self.I(talib.MAX, self.data.High, 20, name='SWING_HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, 20, name='SWING_LOW')
        
        # 🌙 Debug prints for indicator confirmation
        print("✨ Moon Dev Indicators Initialized:")
        print(f"   BBW(20,2) | KIJUN(26) | VOL_MA20 | SWING(20)")
        
    def next(self):
        # 🌙 Moon Dev Themed Debugging
        if len(self.data) % 500 == 0:
            print(f"\n🌕 MOON DEV STATUS UPDATE 🌕")
            print(f"Bar {len(self.data)} | Close: {self.data.Close[-1]:.2f}")
            print(f"BBW: {self.bbw[-1]:.4f} | Percentile: {self.bbw_percentile[-1]:.4f}")
            print(f"Volume: {self.data.Volume[-1]:.2f} vs MA: {self.volume_avg[-1]:.2f}")
        
        # Entry Logic
        if not self.position:
            # Long Entry Conditions
            bbw_cond = self.bbw[-1] < self.bbw_percentile[-1]
            vol_cond = self.data.Volume[-1] > 1.5 * self.volume_avg[-1]
            price_cond = self.data.Close[-1] > self.bb_upper[-1]
            
            if bbw_cond and vol_cond and price_cond:
                # Risk Management Calculation
                sl_price = min(self.swing_low[-1], self.bb_lower[-1])
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = self.data.Close[-1] - sl_price
                
                if risk_per_share > 0:
                    position_size = risk_amount / risk_per_share
                    position_size = int(round(position_size))
                    if position_size > 0:
                        print(f"\n🚀🌙 MOON ROCKET SIGNAL 🚀")
                        print(f"LONG Entry: {self.data.Close[-1]:.2f}")
                        print(f"Size: {position_size} | SL: {sl_price:.2f}")