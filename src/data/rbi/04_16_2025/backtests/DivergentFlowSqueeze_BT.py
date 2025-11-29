```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import pandas_ta as ta

class DivergentFlowSqueeze(Strategy):
    def init(self):
        # Indicator calculations
        # CMF (21-period)
        self.cmf = self.I(lambda high, low, close, volume: ta.cmf(high=high, low=low, close=close, volume=volume, length=21),
                          self.data.High, self.data.Low, self.data.Close, self.data.Volume)
        
        # Price Volume Trend (PVT)
        self.pvt = self.I(lambda close, volume: ta.pvt(close=close, volume=volume), 
                          self.data.Close, self.data.Volume)
        
        # Keltner Channels
        self.ema20 = self.I(talib.EMA, self.data.Close, 20)
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.upper_keltner = self.I(lambda ema, atr: ema + 1.5*atr, self.ema20, self.atr14)
        self.lower_keltner = self.I(lambda ema, atr: ema - 1.5*atr, self.ema20, self.atr14)
        
        # Bollinger Bands
        self.bb_upper, _, self.bb_lower = self.I(talib.BBANDS, self.data.Close, 20, 2, 2)
        
        # Additional indicators
        self.swing_high = self.I(talib.MAX, self.data.High, 20)
        self.swing_low = self.I(talib.MIN, self.data.Low, 20)
        self.volume_ma20 = self.I(talib.SMA, self.data.Volume, 20)
        
        # Risk management variables
        self.consecutive_losses = 0

    def next(self):
        # Skip if not enough data or 3 consecutive losses
        if len(self.data) < 21 or self.consecutive_losses >= 3:
            if self.consecutive_losses >= 3:
                print("🌑 Moon Dev Alert: 3 consecutive losses - trading halted!")
            return
        
        # Current values
        close = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        volume = self.data.Volume[-1]

        # Divergence checks
        cmf_increasing = self.cmf[-3] < self.cmf[-2] < self.cmf[-1]
        pvt_decreasing = self.pvt[-3] > self.pvt[-2] > self.pvt[-1]
        bullish_divergence = cmf_increasing and pvt_decreasing
        
        cmf_decreasing = self.cmf[-3] > self.cmf[-2] > self.cmf[-1]
        pvt_increasing = self.pvt[-3] < self.pvt[-2] < self.pvt[-1]
        bearish_divergence = cmf_decreasing and pvt_increasing

        # Price action checks
        price_lows_decreasing = self.data.Low[-3] > self.data.Low[-2] > self.data.Low[-1]
        price_highs_increasing = self.data.High[-3] < self.data.High[-2] < self.data.High[-1]

        # Keltner width check
        keltner_width = [self.upper_keltner[-i] - self.lower_keltner[-i] for i in range(1,4)]
        keltner_narrowing = keltner_width[0] < keltner_width[1] < keltner_width[2]

        # Volume check
        volume_ok = volume > self.volume_ma20[-1]

        # Entry logic
        if not self.position:
            # Long entry
            if (bullish_divergence and price_lows_decreasing and keltner_narrowing and 
                close > self.swing_high[-1] and volume_ok):
                risk_amount = self.equity * 0.01
                sl_price = self.lower_keltner[-1]
                risk_per_share = close - sl_price
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    self.buy(size=size, sl=sl_price)
                    print(f"🚀 Moon Dev LONG Entry: {close:.2f}, Size: {size}, SL: {sl_price:.2f}")
            
            # Short entry
            elif (bearish_div