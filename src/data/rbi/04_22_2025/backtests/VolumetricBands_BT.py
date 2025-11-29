```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

class VolumetricBands(Strategy):
    risk_percent = 0.01
    
    def init(self):
        typical_price = (self.data.High + self.data.Low + self.data.Close) / 3
        tpv = typical_price * self.data.Volume
        sum_tpv = self.I(talib.SMA, tpv, timeperiod=20, name='SMA_TPV')
        sum_vol = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='SMA_VOL')
        self.vwap = self.I(lambda x, y: x / y, sum_tpv, sum_vol, name='VWAP')
        self.vwap_slope = self.I(talib.LINEARREG_SLOPE, self.vwap, timeperiod=20, name='VWAP_SLOPE')
        
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name='BB_UP', index=0)
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name='BB_LO', index=2)
        
    def next(self):
        if len(self.data.Close) < 20:
            return
        
        current_close = self.data.Close[-1]
        current_upper = self.bb_upper[-1]
        current_lower = self.bb_lower[-1]
        current_vwap = self.vwap[-1]
        vwap_slope = self.vwap_slope[-1]
        prev_vwap_slope = self.vwap_slope[-2] if len(self.vwap_slope) >= 2 else 0
        
        if not self.position:
            if self._band_expanding() and current_close > current_upper and vwap_slope > 0 and prev_vwap_slope > 0:
                self._enter_long(current_close, current_upper, current_vwap)
            elif self._band_expanding() and current_close < current_lower and vwap_slope < 0 and prev_vwap_slope < 0:
                self._enter_short(current_close, current_lower, current_vwap)
        else:
            self._manage_exits(current_vwap)
            self._adjust_stops(current_close)
    
    def _band_expanding(self):
        if len(self.bb_upper) < 2 or len(self.bb_lower) < 2:
            return False
        current_width = self.bb_upper[-1] - self.bb_lower[-1]
        prev_width = self.bb_upper[-2] - self.bb_lower[-2]
        return current_width > prev_width
    
    def _enter_long(self, price, upper, vwap):
        risk_amount = self.equity * self.risk_percent
        stop_price = (upper + vwap) / 2
        risk_per_unit = price - stop_price
        if risk_per_unit <= 0:
            return
        size = int(round(risk_amount / risk_per_unit))
        if size > 0:
            self.buy(size=size, sl=stop_price)
            print(f"🌙✨ LONG ENTRY at {price:.2f}, Size: {size}, SL: {stop_price:.2f} 🚀")
    
    def _enter_short(self, price, lower, vwap):
        risk_amount = self.equity * self.risk_percent
        stop_price = (lower + vwap) / 2
        risk_per_unit = stop_price - price
        if risk_per_unit <= 0:
            return
        size = int(round(risk_amount / risk_per_unit))
        if size > 0:
            self.sell(size=size, sl=stop_price)
            print(f"🌙✨ SHORT ENTRY at {price:.2f}, Size: {size}, SL: {stop_price:.2f} 🚀")
    
    def _manage_exits(self, current_vwap):
        if self.position.is_long and self.data.Close[-1] <= current_vwap:
            self.position.close()
            print(f"🌙✨ EXIT LONG at {self.data.Close[-1]:.2f} (VWAP) 🛑")
        elif self.position.is_short and self.data.Close[-1] >= current_vwap:
            self.position.close()
            print(f"