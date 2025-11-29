Here's the complete implementation of the VolmoBreakout strategy for backtesting.py:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

class VolmoBreakout(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    atr_period = 14
    cmo_period = 14
    signal_period = 9
    bb_period = 20
    swing_period = 20

    def init(self):
        # Calculate indicators using TA-Lib
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR')
        self.cmo = self.I(talib.CMO, self.data.Close, self.cmo_period, name='CMO')
        self.signal = self.I(talib.SMA, self.cmo, self.signal_period, name='Signal')
        
        # Calculate Bollinger Band Width
        def calculate_bbw(close):
            upper, middle, lower = talib.BBANDS(close, self.bb_period, 2, 2)
            return (upper - lower) / middle
        self.bbw = self.I(calculate_bbw, self.data.Close, name='BBW')
        
        # Swing highs/lows
        self.recent_high = self.I(talib.MAX, self.data.High, self.swing_period, name='Recent High')
        self.recent_low = self.I(talib.MIN, self.data.Low, self.swing_period, name='Recent Low')
        
        # Track trade extremes for trailing stop
        self.max_high = None
        self.min_low = None

    def next(self):
        if len(self.data) < max(self.atr_period, self.cmo_period, self.bb_period, self.swing_period) + 2:
            return
        
        # Moon Dev debug prints 🌙
        print(f"🌙 CMO: {self.cmo[-1]:.2f} | Signal: {self.signal[-1]:.2f} | BBW: {self.bbw[-1]:.4f} | ATR: {self.atr[-1]:.2f}")

        if not self.position:
            current_atr = self.atr[-1]
            bbw_atr_ratio = self.bbw[-1] < 0.5 * current_atr

            # Long entry conditions
            if (bbw_atr_ratio and
                crossover(self.cmo, self.signal) and
                self.data.Close[-1] > self.recent_high[-2]):
                
                risk_amount = self.equity * self.risk_percent
                sl_distance = current_atr * 1
                position_size = int(round(risk_amount / sl_distance))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=self.data.Close[-1] - sl_distance)
                    self.max_high = self.data.High[-1]
                    print(f"🚀🌙 LONG ENTRY | Size: {position_size} | Entry: {self.data.Close[-1]:.2f} | SL: {self.data.Close[-1] - sl_distance:.2f}")

            # Short entry conditions
            elif (bbw_atr_ratio and
                  crossunder(self.cmo, self.signal) and
                  self.data.Close[-1] < self.recent_low[-2]):
                
                risk_amount = self.equity * self.risk_percent
                sl_distance = current_atr * 1
                position_size = int(round(risk_amount / sl_distance))
                
                if position_size > 0:
                    self.sell(size=position_size, sl=self.data.Close[-1] + sl_distance)
                    self.min_low = self.data.Low[-1]
                    print(f"🌑🌙 SHORT ENTRY | Size: {position_size} | Entry: {self.data.Close[-1]:.2f} | SL: {self.data.Close[-1] + sl_distance:.2f}")

        else:
            if self.position.is_long:
                self.max_high = max(self.max_high, self.data.High[-1])
                trailing_stop = self.max_high - 2 * self.atr[-1]
                
                if crossunder(self.cmo, self.signal):
                    self.position.close()
                    print(f"🌙✨ EXIT LONG | CMO cross under signal | Price: {self.data.Close[-1]:.2f}")
                elif self.data.Low[-1] < trailing_stop: