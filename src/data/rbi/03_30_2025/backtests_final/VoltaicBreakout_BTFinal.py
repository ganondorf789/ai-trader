I'll complete and debug the VoltaicBreakout strategy code while maintaining all the original strategy logic. Here's the fixed version with proper Moon Dev themed debug prints and all technical issues resolved:

```python
# 🌙 Moon Dev's VoltaicBreakout Backtest 🌙
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
import talib

class VoltaicBreakout(Strategy):
    # Strategy parameters
    risk_per_trade = 0.01  # 1% risk per trade
    time_exit_bars = 10    # Bars until time-based exit
    
    def init(self):
        # 🌙 Initialize indicators using TA-Lib
        # Bollinger Bands
        self.inner_upper = self.I(lambda close: talib.BBANDS(close, timeperiod=20, nbdevup=1.5, nbdevdn=1.5)[0], self.data.Close)
        self.inner_lower = self.I(lambda close: talib.BBANDS(close, timeperiod=20, nbdevup=1.5, nbdevdn=1.5)[2], self.data.Close)
        self.outer_upper = self.I(lambda close: talib.BBANDS(close, timeperiod=20, nbdevup=2.5, nbdevdn=2.5)[0], self.data.Close)
        self.outer_lower = self.I(lambda close: talib.BBANDS(close, timeperiod=20, nbdevup=2.5, nbdevdn=2.5)[2], self.data.Close)
        
        # Volume MA and ATR
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        print("🌙 Lunar indicators initialized! Ready for launch 🚀")

    def next(self):
        # 🌙 Skip initial bars without indicator data
        if len(self.data) < 20:
            return
        
        # 🌙 Entry Logic
        if not self.position:
            # Volatility filter check
            volatility_ok = (self.outer_upper[-1] - self.outer_lower[-1]) >= 2 * self.atr[-1]
            
            # Long entry: Close above outer upper + volume surge
            if (self.data.Close[-1] > self.outer_upper[-1] and 
                self.data.Volume[-1] > self.volume_ma[-1] and 
                volatility_ok):
                sl_price = self.outer_lower[-1]
                risk_per_share = self.data.Close[-1] - sl_price
                if risk_per_share > 0:
                    size = int(round((self.risk_per_trade * self.equity) / risk_per_share))
                    self.buy(size=size, sl=sl_price, tag="Moon Long Entry 🌕")
                    print(f"🚀 LIFTOFF! Long at {self.data.Close[-1]:.2f} | Size: {size} | SL: {sl_price:.2f}")
            
            # Short entry: Close below outer lower + volume surge
            elif (self.data.Close[-1] < self.outer_lower[-1] and 
                  self.data.Volume[-1] > self.volume_ma[-1] and 
                  volatility_ok):
                sl_price = self.outer_upper[-1]
                risk_per_share = sl_price - self.data.Close[-1]
                if risk_per_share > 0:
                    size = int(round((self.risk_per_trade * self.equity) / risk_per_share))
                    self.sell(size=size, sl=sl_price, tag="Moon Short Entry 🌑")
                    print(f"🌑 BLACK HOLE! Short at {self.data.Close[-1]:.2f} | Size: {size} | SL: {sl_price:.2f}")
        
        # 🌙 Exit Logic
        else:
            if self.position.is_long:
                # Retracement exit
                if self.data.Close[-1] < self.inner_upper[-1]:
                    self.position.close()
                    print(f"🌙✨ Moon Gravity Pull! Closing long at {self.data.Close[-1]:.2f}")
                # Time-based exit
                elif len(self.data) - self.position.entry_bar >= self.time_exit_bars:
                    self.position.close()
                    print(f"⏳🚀 Time Warp Exit! Closing long after {self.time_exit_bars} bars")
            elif self.position.is_short:
                # Retracement exit
                if self.data.Close[-1] > self.inner_l