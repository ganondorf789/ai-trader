Here's the complete fixed code with Moon Dev themed debug prints and proper position sizing:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class VoltaicBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    atr_multiplier_sl = 3  # Initial stop loss multiplier
    consolidation_period = 5  # Bars required for consolidation
    
    def init(self):
        # Calculate indicators using TA-Lib with self.I()
        self.ma20 = self.I(talib.SMA, self.data.Close, timeperiod=20, name='MA_20')
        self.atr20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=20, name='ATR_20')
        self.volume_slope = self.I(talib.LINEARREG_SLOPE, self.data.Volume, timeperiod=5, name='Volume_Slope')
        
        print("🌙 VoltaicBreakout Strategy Initialized! ✨")
        print("✨ All indicators powered by TA-Lib with Moon Dev precision! 🌙")
        
    def next(self):
        current_close = self.data.Close[-1]
        current_ma = self.ma20[-1]
        current_atr = self.atr20[-1]
        
        # Calculate dynamic bands
        upper_band = current_ma + 2 * current_atr
        lower_band = current_ma - 2 * current_atr
        exit_upper = current_ma + current_atr
        exit_lower = current_ma - current_atr
        
        # Moon Dev debug prints
        print(f"🌙 Bar {len(self.data)-1}: Close={current_close:.2f}, MA={current_ma:.2f}, ATR={current_atr:.2f}")
        print(f"🌙 Bands: Upper={upper_band:.2f}, Lower={lower_band:.2f}")
        
        # Check for entry signals
        if not self.position:
            # Long entry conditions
            if current_close > upper_band:
                if self._valid_consolidation(-1, direction="long") and self.volume_slope[-2] < 0:
                    self._enter_long(current_close, current_atr)
            
            # Short entry conditions
            elif current_close < lower_band:
                if self._valid_consolidation(-1, direction="short") and self.volume_slope[-2] < 0:
                    self._enter_short(current_close, current_atr)
        
        # Manage open positions
        else:
            if self.position.is_long:
                if self.data.Close[-1] <= exit_upper:
                    self.position.close()
                    print(f"✨ Closing LONG at {self.data.Close[-1]:.2f} (Hit Exit Band)")
                
                # Update trailing stop
                elif self.data.High[-1] > self.position.entry_price + current_atr:
                    new_sl = self.position.entry_price + current_atr * (self.atr_multiplier_sl - 2)
                    self.position.sl = max(self.position.sl or 0, new_sl)
                    
            elif self.position.is_short:
                if self.data.Close[-1] >= exit_lower:
                    self.position.close()
                    print(f"✨ Closing SHORT at {self.data.Close[-1]:.2f} (Hit Exit Band)")
                
                # Update trailing stop
                elif self.data.Low[-1] < self.position.entry_price - current_atr:
                    new_sl = self.position.entry_price - current_atr * (self.atr_multiplier_sl - 2)
                    self.position.sl = min(self.position.sl or float('inf'), new_sl)

    def _valid_consolidation(self, current_idx, direction):
        # Check previous 5 bars were within their bands
        for i in range(current_idx-1, current_idx-self.consolidation_period-1, -1):
            try:
                ma = self.ma20[i]
                atr = self.atr20[i]
                close = self.data.Close[i]
                
                if direction == "long":
                    if close > (ma + 2*atr) or close < (ma - 2*atr):
                        return False
                else:
                    if close < (ma - 2*atr) or close > (ma + 2*atr):
                        return False
            except IndexError:
                return False
        return True

    def _enter_long(self, entry_price, atr):
        risk_amount = self.risk_per_t