Here's the fixed code with all backtesting.lib references removed and proper indicator usage:

```python
import pandas as pd
import talib
from backtesting import Strategy, Backtest
import numpy as np

class ChannelSurgeTrail(Strategy):
    risk_per_trade = 0.01
    max_consecutive_losses = 5
    atr_multiplier = 3
    adx_threshold = 25
    adx_exit_threshold = 20
    donchian_period = 20
    adx_period = 14
    atr_period = 14
    
    def init(self):
        self.consecutive_losses = 0
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.upper_band = self.I(talib.MAX, self.data.High, timeperiod=self.donchian_period)
        self.lower_band = self.I(talib.MIN, self.data.Low, timeperiod=self.donchian_period)
        
    def next(self):
        if len(self.data.Close) < max(self.donchian_period, self.adx_period, self.atr_period) + 1:
            return
            
        current_adx = self.adx[-1]
        current_atr = self.atr[-1]
        
        if not self.position:
            if self.data.Close[-1] > self.upper_band[-1] and current_adx > self.adx_threshold:
                risk_amount = self.risk_per_trade * self.equity
                position_size = int(round(risk_amount / (self.atr_multiplier * current_atr)))
                if position_size == 0:
                    return
                sl = self.data.Close[-1] - self.atr_multiplier * current_atr
                self.buy(size=position_size, sl=sl)
                self.position.data['extreme'] = self.data.High[-1]
                print(f"🌙 MOON DEV LONG ENTRY! ✨ Size: {position_size} | Entry: {self.data.Close[-1]:.2f} | SL: {sl:.2f}")
                
            elif self.data.Close[-1] < self.lower_band[-1] and current_adx > self.adx_threshold:
                risk_amount = self.risk_per_trade * self.equity
                position_size = int(round(risk_amount / (self.atr_multiplier * current_atr)))
                if position_size == 0:
                    return
                sl = self.data.Close[-1] + self.atr_multiplier * current_atr
                self.sell(size=position_size, sl=sl)
                self.position.data['extreme'] = self.data.Low[-1]
                print(f"🌙 MOON DEV SHORT ENTRY! ✨ Size: {position_size} | Entry: {self.data.Close[-1]:.2f} | SL: {sl:.2f}")
        else:
            if self.position.is_long:
                self.position.data['extreme'] = max(self.data.High[-1], self.position.data['extreme'])
                new_sl = self.position.data['extreme'] - self.atr_multiplier * current_atr
                self.position.sl = max(new_sl, self.position.sl or 0)
                
                if self.data.Low[-1] <= self.position.sl:
                    print(f"🚀 MOON TRAIL STOP HIT! | Exit: {self.position.sl:.2f} | PnL: {self.position.pl:.2f}")
                    self.position.close()
                elif current_adx < self.adx_exit_threshold:
                    print(f"⚠️ MOON EMERGENCY EXIT! | ADX: {current_adx:.2f} | PnL: {self.position.pl:.2f}")
                    self.position.close()
                    
            elif self.position.is_short:
                self.position.data['extreme'] = min(self.data.Low[-1], self.position.data['extreme'])
                new_sl = self.position.data['extreme'] + self.atr_multiplier * current_atr
                self.position.sl = min(new_sl, self.position.sl or np.inf)
                
                if self.data.High[-1] >= self.position.sl:
                    print(f"🚀 MOON TRAIL STOP HIT! | Exit: {self.position.sl:.2f} | PnL: {self.position.pl:.2f}")
                    self.position.close