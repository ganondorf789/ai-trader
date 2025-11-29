I'll fix the code while maintaining the original strategy logic. Here's the debugged version with Moon Dev themed improvements:

```python
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

class VolumetricDivergence(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Clean column names and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col], errors='ignore')
        
        # Indicator calculations
        # Volume MA for filtering
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20)
        
        # VW-MACD Components
        self.vwap_50 = self.I(lambda h, l, c, v: ta.vwap(high=h, low=l, close=c, volume=v, length=50),
                             self.data.High, self.data.Low, self.data.Close, self.data.Volume)
        self.macd_line = self.I(lambda s: talib.MACD(s, 12, 26, 9)[0], self.vwap_50)
        
        # Chande Momentum Oscillator
        self.cmo = self.I(talib.CMO, self.data.Close, 14)
        
        # Bollinger Bandwidth
        self.upper_band = self.I(lambda c: talib.BBANDS(c, 20, 2, 2)[0], self.data.Close)
        self.lower_band = self.I(lambda c: talib.BBANDS(c, 20, 2, 2)[2], self.data.Close)
        self.bandwidth = self.I(lambda u, l: u - l, self.upper_band, self.lower_band)
        
        # Swing levels for SL
        self.swing_high = self.I(talib.MAX, self.data.High, 20)
        self.swing_low = self.I(talib.MIN, self.data.Low, 20)
        
        # Volatility measure
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
        print("🌙 Lunar Indicators Activated! Ready for launch 🚀")

    def next(self):
        # Moon Dev debug pulse
        if len(self.data) % 100 == 0:
            print(f"🌙 Processing Bar {len(self.data)}: Close={self.data.Close[-1]:.2f} ✨")
        
        if len(self.data) < 50:
            return
        
        # Entry Logic Core - Manual crossover implementation
        macd_cross_up = self.macd_line[-2] < 0 and self.macd_line[-1] > 0
        macd_cross_down = self.macd_line[-2] > 0 and self.macd_line[-1] < 0
        
        # Divergence detection
        price_lower_low = self.data.Low[-1] < self.data.Low[-2]
        cmo_higher_low = self.cmo[-1] > self.cmo[-2]
        bullish_div = price_lower_low and cmo_higher_low
        
        price_higher_high = self.data.High[-1] > self.data.High[-2]
        cmo_lower_high = self.cmo[-1] < self.cmo[-2]
        bearish_div = price_higher_high and cmo_lower_high
        
        # Volume filter
        volume_ok = self.data.Volume[-1] > self.volume_ma[-1]
        
        # Entry Execution
        if not self.position and volume_ok:
            if macd_cross_up and bullish_div:
                self.enter_long()
            elif macd_cross_down and bearish_div:
                self.enter_short()
        
        # Exit Logic
        if self.position:
            if len(self.bandwidth) >= 3:
                bandwidth_contraction = (self.bandwidth[-1] < self.bandwidth[-2]) and \
                                       (self.bandwidth[-2] < self.bandwidth[-3])
                if bandwidth_contraction:
                    self.position.close()
                    print(f"🌙 Bandwidth Contraction! Closing {round(self.position.size)} shares ✨")

    def enter_long(self):
        entry_price = self.data.Close[-1]
        swing_low = self.swing_low[-1]
        atr = self.atr[-1]
        
        # Position sizing -