Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations:

```python
# 🌙 MOON DEV BACKTESTING SCRIPT - VOLUMETRIC BREAKOUT STRATEGY 🚀

import pandas as pd
import talib
from backtesting import Strategy, Backtest

class VolumetricBreakout(Strategy):
    risk_per_trade = 0.01  # 1% of equity per trade 🌕
    
    def init(self):
        # 🌟 INDICATOR CALCULATIONS
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume)
        
        # Price and OBV swing highs (20-period)
        self.price_swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='Price Swing High 🌙')
        self.obv_swing_high = self.I(talib.MAX, self.obv, timeperiod=20, name='OBV Swing High 📈')
        
        # Bollinger Bands components
        self.bb_upper = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0], 
                              self.data.Close, name='BB Upper 🎢')
        self.bb_middle = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1], 
                               self.data.Close, name='BB Middle 📊')
        self.bb_lower = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2], 
                              self.data.Close, name='BB Lower 🎢')
        self.bb_width = self.I(lambda u, l, m: (u - l)/m, 
                              self.bb_upper, self.bb_lower, self.bb_middle, name='BB Width 🌊')
        self.bb_width_min = self.I(talib.MIN, self.bb_width, timeperiod=20, name='BB Width Min 🔍')

    def next(self):
        # 🛑 Skip early bars without indicator data
        if len(self.adx) < 20 or len(self.obv_swing_high) < 2:
            return
            
        # 🌙 ENTRY LOGIC: Bearish Divergence Detection
        if not self.position:
            # Price makes new high while OBV fails to confirm
            price_divergence = (self.price_swing_high[-1] > self.price_swing_high[-2])
            obv_divergence = (self.obv_swing_high[-1] < self.obv_swing_high[-2])
            adx_valid = self.adx[-1] > 25
            
            if price_divergence and obv_divergence and adx_valid:
                # 🎯 RISK MANAGEMENT CALCULATIONS
                entry_price = self.data.Open[-1]  # Next candle's open
                stop_loss_price = self.price_swing_high[-1]  # Recent swing high
                risk_per_share = stop_loss_price - entry_price
                
                if risk_per_share > 0:
                    position_size = int(round((self.equity * self.risk_per_trade) / risk_per_share))
                    if position_size > 0:
                        self.sell(size=position_size, sl=stop_loss_price, tag='Short Entry 🌌')
                        print(f"🚀 MOON SHOT! Short {position_size} @ {entry_price:.2f} | SL: {stop_loss_price:.2f} | Risk: {risk_per_share*position_size:.2f}")

        # 💫 EXIT LOGIC
        else:
            # Volatility collapse or trend weakening
            volatility_exit = self.bb_width[-1] <= self.bb_width_min[-1]
            trend_exit = self.adx[-1] < 20
            
            if volatility_exit or trend_exit:
                self.position.close()
                print(f"🌑 MOON OUT! Closed @ {self.data.Close[-1]:.2f} | Reason: {'Volatility' if volatility_exit else 'Trend'}")

# 📂 DATA PREPARATION
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15