Here's the fully debugged and Moon Dev optimized backtest code for the LiquiBand Breakout Strategy:

```python
# 🌙 Moon Dev Backtest AI Implementation: LiquiBand Breakout Strategy ✨

import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

class LiquiBandStrategy(Strategy):
    risk_percent = 0.01  # 1% risk per trade (fixed as fraction)
    
    def init(self):
        # 🌙 Prepare indicator data using TA-Lib
        close = self.data.Close.values.astype(float)
        volume = self.data.Volume.values.astype(float)
        
        # 🎯 Bollinger Bands (20,2)
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        self.upper_band = self.I(lambda: upper, name='Upper BB')
        self.lower_band = self.I(lambda: lower, name='Lower BB')
        
        # 📉 Bollinger Band Width calculation
        bb_width = (upper - lower) / middle
        self.bb_width = self.I(lambda: bb_width, name='BB Width')  # Store as indicator
        self.bb_width_ma = self.I(talib.SMA, bb_width, timeperiod=20, name='BB Width MA')
        
        # 💧 Liquidation Cluster Proxy (Volume MA)
        self.volume_ma = self.I(talib.SMA, volume, timeperiod=20, name='Volume MA')
        
        print("🌙✨ Moon Dev Indicators Initialized! BBANDS, Width MA & Volume MA Ready 🚀")

    def next(self):
        # ⏳ Skip first 20 periods for indicator warmup
        if len(self.data) < 20:
            return
            
        # 📊 Current market conditions
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        bb_contraction = (self.bb_width[-1] < self.bb_width_ma[-1])
        liquidation_zone = (current_volume > self.volume_ma[-1])
        
        # 🚦 Entry Conditions
        if not self.position:
            # 🌕 Long Entry: Close > Upper BB + Contraction + Liquidation Cluster
            if bb_contraction and liquidation_zone and current_close > self.upper_band[-1]:
                risk_amount = self.equity * self.risk_percent
                entry_price = self.data.Close[-1]  # Use current close for entry
                stop_loss = min(self.data.Low[-1], self.lower_band[-1])  # More conservative SL
                
                if (risk_per_share := entry_price - stop_loss) > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:  # Ensure valid position size
                        self.buy(size=position_size, sl=stop_loss)
                        print(f"🚀🌙 MOON DEV LONG SIGNAL! Size: {position_size} | Entry: {entry_price:.2f} | SL: {stop_loss:.2f}")
            
            # 🌑 Short Entry: Close < Lower BB + Contraction + Liquidation Cluster
            elif bb_contraction and liquidation_zone and current_close < self.lower_band[-1]:
                risk_amount = self.equity * self.risk_percent
                entry_price = self.data.Close[-1]  # Use current close for entry
                stop_loss = max(self.data.High[-1], self.upper_band[-1])  # More conservative SL
                
                if (risk_per_share := stop_loss - entry_price) > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:  # Ensure valid position size
                        self.sell(size=position_size, sl=stop_loss)
                        print(f"🌑🌙 MOON DEV SHORT SIGNAL! Size: {position_size} | Entry: {entry_price:.2f} | SL: {stop_loss:.2f}")

        # 💸 Exit Conditions
        else:
            # ✨ Long Exit: Price touches Lower BB
            if self.position.is_long and self.data.Low[-1] <= self.lower_band[-1]:
                self.position.close()
                print(f"✅🌙 MOON DEV LONG EXIT: Price {self.data.Close[-1]:.2f} touched Lower BB!")
            
            # ✨ Short Exit: Price touches Upper BB
            elif self.position.is_short and self.data.High[-1] >= self.upper_band[-1]:
                self.position.close()
                print