Here's the fixed code with all technical issues resolved while preserving the original strategy logic:

```python
import pandas as pd
import talib
import pandas_ta as pta
from backtesting import Backtest, Strategy

class VortexFlowExtension(Strategy):
    initial_capital = 1_000_000
    risk_per_trade = 0.01  # 1% per trade
    
    def init(self):
        # Precompute all indicators
        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        volume = self.data.Volume
        
        # Vortex Indicator
        vi = pta.vortex(high=high, low=low, close=close, length=14)
        self.vi_plus = self.I(lambda: vi['VORTIC_14_1'], name='VI_plus')
        self.vi_minus = self.I(lambda: vi['VORTIC_14_2'], name='VI_minus')
        
        # Chaikin Money Flow
        self.cmf = self.I(pta.cmf, high=high, low=low, close=close, volume=volume, length=21, name='CMF')
        
        # ATR and its SMA
        atr = talib.ATR(high, low, close, timeperiod=14)
        self.atr_sma = self.I(talib.SMA, atr, timeperiod=20, name='ATR_SMA')
        self.I(lambda: atr, name='ATR')
        
        # Swing High/Low
        self.swing_high = self.I(talib.MAX, high, timeperiod=20, name='Swing_High')
        self.swing_low = self.I(talib.MIN, low, timeperiod=20, name='Swing_Low')
    
    def next(self):
        # Skip if not enough data
        if len(self.data) < 20:
            return
        
        # Check emergency exits first
        for trade in self.trades:
            price = self.data.Close[-1]
            if trade.is_long:
                if (self.vi_minus[-2] < self.vi_plus[-2] and self.vi_minus[-1] > self.vi_plus[-1]):  # Bearish crossover
                    print(f"🌙🚨 EMERGENCY EXIT LONG @ {price:.2f} | Moon Shield Engaged! 🛡️")
                    trade.close()
            else:
                if (self.vi_plus[-2] < self.vi_minus[-2] and self.vi_plus[-1] > self.vi_minus[-1]):  # Bullish crossover
                    print(f"🌙🚨 EMERGENCY EXIT SHORT @ {price:.2f} | Moon Shield Engaged! 🛡️")
                    trade.close()
        
        # Check existing positions
        if len(self.trades) >= 5:
            return
        
        # Long entry conditions
        atr_condition = self.data.ATR[-1] < self.data.ATR_SMA[-1]
        vi_cross = (self.vi_plus[-2] < self.vi_minus[-2] and self.vi_plus[-1] > self.vi_minus[-1])  # Bullish crossover
        cmf_condition = self.data.CMF[-1] > 0
        
        if atr_condition and vi_cross and cmf_condition:
            entry_price = self.data.Open[-1]
            swing_low = self.swing_low[-1]
            swing_high = self.swing_high[-1]
            risk = entry_price - swing_low
            
            if risk <= 0:
                return
            
            position_size = (self.risk_per_trade * self.equity) / risk
            position_size = int(round(position_size))
            
            if position_size > 0:
                swing_range = swing_high - swing_low
                tp1 = entry_price + swing_range * 1.27
                tp2 = entry_price + swing_range * 1.618
                
                half = position_size // 2
                rem = position_size - half
                
                self.buy(size=half, sl=swing_low, tp=tp1, tag='Long_TP1')
                self.buy(size=rem, sl=swing_low, tp=tp2, tag='Long_TP2')
                print(f"🌙✨ LONG ENTRY @ {entry_price:.2f} | Size: {position_size} | Moon Power Activated! 🚀")
        
        # Short entry conditions
        vi_cross_short = (self.vi_minus[-2] < self.vi_plus[-2] and self.