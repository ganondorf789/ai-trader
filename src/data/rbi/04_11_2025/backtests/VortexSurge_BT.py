```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import pandas_ta as ta
import talib

class VortexSurge(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    adx_threshold = 25
    exit_adx = 20
    volume_multiplier = 1.2
    atr_multiplier = 2
    
    def init(self):
        # Calculate indicators with proper self.I() wrapping
        self.vi_plus = self.I(self._calculate_vi_plus, self.data.High, self.data.Low, self.data.Close)
        self.vi_minus = self.I(self._calculate_vi_minus, self.data.High, self.data.Low, self.data.Close)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, 14)
        self.vol_ma = self.I(talib.SMA, self.data.Volume, 20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
    def _calculate_vi_plus(self, high, low, close):
        """Calculate Vortex Indicator (VI+) using pandas_ta"""
        vi = ta.vi(high=high, low=low, close=close, length=14)
        return vi['VIPT']
    
    def _calculate_vi_minus(self, high, low, close):
        """Calculate Vortex Indicator (VI-) using pandas_ta"""
        vi = ta.vi(high=high, low=low, close=close, length=14)
        return vi['VIMN']

    def next(self):
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        
        # Moon Dev debug prints 🌙
        if len(self.data) % 100 == 0:
            print(f"\n🌙 Moon Dev Status Update @ {self.data.index[-1]} ✨")
            print(f"   Close: {current_close:.2f}, VI+:{self.vi_plus[-1]:.2f}, VI-:{self.vi_minus[-1]:.2f}")
            print(f"   ADX: {self.adx[-1]:.2f}, Vol Ratio: {current_volume/self.vol_ma[-1]:.2f}x")

        if not self.position:
            # Long entry condition
            if (crossover(self.vi_plus, self.vi_minus) and
                self.adx[-1] > self.adx_threshold and
                current_volume > self.vol_ma[-1] * self.volume_multiplier):
                
                self._enter_trade('long', current_close)
            
            # Short entry condition
            elif (crossover(self.vi_minus, self.vi_plus) and
                  self.adx[-1] > self.adx_threshold and
                  current_volume > self.vol_ma[-1] * self.volume_multiplier):
                
                self._enter_trade('short', current_close)
        
        else:
            # Exit condition for both long/short
            if self.adx[-1] < self.exit_adx:
                self.position.close()
                print(f"🌙🌊 ADX Collapse! Closing @ {current_close:.2f} (ADX: {self.adx[-1]:.2f})")

    def _enter_trade(self, direction, entry_price):
        """Moon Dev's Unified Trade Entry Logic 🚀"""
        atr_value = self.atr[-1]
        stop_loss = atr_value * self.atr_multiplier
        
        # Calculate position size with proper rounding
        risk_amount = self.equity * self.risk_percent
        position_size = risk_amount / (stop_loss if direction == 'long' else -stop_loss)
        position_size = int(round(position_size))
        
        if direction == 'long':
            sl_price = entry_price - stop_loss
            self.buy(size=position_size, sl=sl_price)
            print(f"\n🚀🌙 LONG SURGE! {position_size} units @ {entry_price:.2f}")
            print(f"   SL: {sl_price:.2f} ({stop_loss:.2f} risk)")
        else:
            sl_price = entry_price + stop_loss
            self.sell(size=position_size, sl=sl_price)
            print(f"\n🚀🌙 SHORT VORTEX! {position_size} units @ {entry_price:.2f}")
            print(f"   SL: {sl_price:.2f