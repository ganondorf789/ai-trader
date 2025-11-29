Here's the complete fixed code with Moon Dev themed debug prints and proper position sizing:

```python
# 🌙 Moon Dev's BandwidthBreakout Backtest 🌙
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

class BandwidthBreakout(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    bb_period = 20
    volume_ma_period = 10
    
    def init(self):
        # 🌙 Calculate Bollinger Bands components
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, 
                              timeperiod=self.bb_period, nbdevup=2, nbdevdn=2, matype=0, name='BB_UPPER', which=0)
        self.bb_middle = self.I(talib.BBANDS, self.data.Close,
                               timeperiod=self.bb_period, nbdevup=2, nbdevdn=2, matype=0, name='BB_MIDDLE', which=1)
        self.bb_lower = self.I(talib.BBANDS, self.data.Close,
                              timeperiod=self.bb_period, nbdevup=2, nbdevdn=2, matype=0, name='BB_LOWER', which=2)
        
        # 🌙 Calculate Bollinger Band Width
        self.bb_width = self.I(lambda u, m, l: (u-l)/m,
                              self.bb_upper, self.bb_middle, self.bb_lower, name='BB_WIDTH')
        
        # 🌙 Calculate 20-period low of BB Width
        self.bb_width_low = self.I(talib.MIN, self.bb_width, timeperiod=self.bb_period, name='BB_WIDTH_LOW')
        
        # 🌙 Volume indicators
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_ma_period, name='VOLUME_MA')
        
        self.entry_bar = 0  # Track entry bar for time-based exit

    def next(self):
        # 🌙✨ Skip initial bars without indicator values
        if len(self.data) < self.bb_period or np.isnan(self.bb_width_low[-1]):
            print("🌙 Waiting for indicators to warm up...")
            return

        # 🌙 Current indicator values
        price_close = self.data.Close[-1]
        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]
        volume_spike = self.data.Volume[-1] >= 1.5 * self.volume_ma[-1]
        bb_contraction = self.bb_width[-1] <= self.bb_width_low[-1]

        # 🌙✨ Entry Logic
        if not self.position:
            if bb_contraction and volume_spike:
                # Long Entry 🚀
                if price_close > bb_upper:
                    risk = price_close - bb_lower
                    if risk <= 0: 
                        print("🌑 Risk calculation failed - skipping trade")
                        return
                    position_size = int(round((self.risk_pct * self.equity) / risk))
                    if position_size > 0:
                        self.buy(size=position_size, sl=bb_lower)
                        self.entry_bar = len(self.data)
                        print(f"🚀🌙 MOON SHOT! LONG Signal Activated | Size: {position_size} | Risk: {risk:.2f}")

                # Short Entry 🌑
                elif price_close < bb_lower:
                    risk = bb_upper - price_close
                    if risk <= 0: 
                        print("🌑 Risk calculation failed - skipping trade")
                        return
                    position_size = int(round((self.risk_pct * self.equity) / risk))
                    if position_size > 0:
                        self.sell(size=position_size, sl=bb_upper)
                        self.entry_bar = len(self.data)
                        print(f"🌑🌙 DARK SIDE! SHORT Signal Activated | Size: {position_size} | Risk: {risk:.2f}")

        # 🌙 Exit Logic
        else:
            bars_since_entry = len(self.data) - self.entry_bar
            
            if self.position.is_long:
                # Stop loss or opposite band touch
                if self.data.Low[-1] <= bb_lower or bars_since_entry >= 5:
                    self.position.close()
                    print(f"🌙✨ Closing LONG: {'Band Touch Exit' if self.data.Low[-1] <= bb_lower else 'Time Exit'}")

            elif self.position.is_short:
                # Stop