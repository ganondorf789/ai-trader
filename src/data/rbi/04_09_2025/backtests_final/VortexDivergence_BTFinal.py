I'll fix the incomplete code while maintaining the original strategy logic. Here's the debugged version with Moon Dev themed improvements:

```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

class VortexDivergence(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    swing_period = 5     # Swing detection window
    
    def init(self):
        # Moon Dev Initialization Phase 🌙
        print("🌙 Initializing Vortex Divergence Strategy...")
        
        # Vortex Indicator
        vi = ta.vortex(high=self.data.High, low=self.data.Low, close=self.data.Close, length=14)
        self.vi_plus = self.I(lambda: vi['VIPT_14'], name='VI+')
        self.vi_minus = self.I(lambda: vi['VIMT_14'], name='VI-')
        
        # Chande Momentum Oscillator
        self.cmo = self.I(talib.CMO, self.data.Close, timeperiod=14)
        
        # Swing detection
        self.price_swing_low = self.I(talib.MIN, self.data.Low, self.swing_period)
        self.price_swing_high = self.I(talib.MAX, self.data.High, self.swing_period)
        self.cmo_swing_low = self.I(talib.MIN, self.cmo, self.swing_period)
        self.cmo_swing_high = self.I(talib.MAX, self.cmo, self.swing_period)
        
        # EMA Bands
        self.ema = self.I(talib.EMA, self.data.Close, 20)
        self.prev_swing_low = self.prev_swing_high = None
        print("✨ Indicators successfully initialized!")

    def next(self):
        current_close = self.data.Close[-1]
        
        # Calculate dynamic EMA bands
        ema_value = self.ema[-1]
        ema_upper = ema_value * 1.02
        ema_lower = ema_value * 0.98
        
        # Moon Dev Risk Management 🌙
        equity = self.equity
        position_size = 0
        
        # Long Entry Logic
        if (self.vi_minus[-1] > self.vi_plus[-1] and  # VI- > VI+
            self.data.Low[-1] == self.price_swing_low[-1] and  # New swing low
            self.prev_swing_low and  # Previous swing exists
            (self.data.Low[-1] < self.prev_swing_low[0]) and  # Lower low
            (self.cmo_swing_low[-1] > self.prev_swing_low[1]) and  # CMO higher low
            (self.data.Volume[-1] < self.data.Volume[self.prev_swing_low[2]])):  # Volume decrease
            
            risk_amount = equity * self.risk_percent
            entry_price = self.data.Open[-1]
            stop_loss = self.data.Low[-1] * 0.999
            risk_per_unit = entry_price - stop_loss
            
            if risk_per_unit > 0:
                position_size = int(round(risk_amount / risk_per_unit))
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_loss)
                    print(f"🌙✨🚀 MOON DEV LONG SIGNAL! Entry: {entry_price:.2f}, Size: {position_size}, SL: {stop_loss:.2f}")
            
            self.prev_swing_low = (self.data.Low[-1], self.cmo_swing_low[-1], len(self.data)-1)

        # Short Entry Logic
        elif (self.vi_plus[-1] > self.vi_minus[-1] and  # VI+ > VI-
              self.data.High[-1] == self.price_swing_high[-1] and  # New swing high
              self.prev_swing_high and  # Previous swing exists
              (self.data.High[-1] > self.prev_swing_high[0]) and  # Higher high
              (self.cmo_swing_high[-1] < self.prev_swing_high[1]) and  # CMO lower high
              (self.data.Volume[-1] < self.data.Volume[self.prev_swing_high[2]])):  # Volume decrease
            
            risk_amount = equity * self.risk_percent
            entry_price = self.data.Open[-1]
            stop_loss = self.data.High[-1] *