Here's the complete backtesting implementation for the VolumetricVortex strategy:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION 🌙
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import pandas_ta as ta
import talib

class VolumetricVortex(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    lvn_period = 20 * 96  # 20 sessions (96 periods/day for 15m)
    vortex_period = 14
    kroll_period = 10
    kroll_multiplier = 3
    
    def init(self):
        # 🌙 Data preparation
        self.data.df.index = pd.to_datetime(self.data.df.index)
        
        # 🌟 Indicator Calculation
        # Volume Profile (Low Volume Node approximation)
        self.low_volume_node = self.I(talib.MIN, self.data.Close, self.lvn_period)
        
        # Vortex Indicator
        self.vi_plus, self.vi_minus = self.I(ta.vortex,
                                            high=self.data.High,
                                            low=self.data.Low,
                                            close=self.data.Close,
                                            length=self.vortex_period,
                                            drift=1,
                                            append=False)
        
        # Chande's Kroll Volatility Bands
        def calc_upper_band(high, low, close):
            hh = talib.MAX(high, self.kroll_period)
            atr = talib.ATR(high, low, close, self.kroll_period)
            return hh + self.kroll_multiplier * atr
        
        def calc_lower_band(high, low, close):
            ll = talib.MIN(low, self.kroll_period)
            atr = talib.ATR(high, low, close, self.kroll_period)
            return ll - self.kroll_multiplier * atr
        
        self.upper_band = self.I(calc_upper_band, self.data.High, self.data.Low, self.data.Close)
        self.lower_band = self.I(calc_lower_band, self.data.High, self.data.Low, self.data.Close)
        
        print("🌙 VolumetricVortex Strategy Initialized! 🚀")

    def next(self):
        price = self.data.Close[-1]
        
        # 🌙 Trade Entry Logic
        if not self.position:
            # Long Entry Conditions
            long_trigger = (price > self.low_volume_node[-1] and 
                           crossover(self.vi_plus, self.vi_minus))
            
            # Short Entry Conditions
            short_trigger = (price < self.low_volume_node[-1] and 
                            crossover(self.vi_minus, self.vi_plus))
            
            if long_trigger:
                self.enter_long(price)
            
            elif short_trigger:
                self.enter_short(price)
        
        # 🌙 Trailing Stop Management
        else:
            self.update_trailing_stops()

    def enter_long(self, price):
        sl = self.low_volume_node[-1]
        risk = price - sl
        if risk <= 0: return
        
        position_size = self.calculate_position_size(risk)
        if position_size > 0:
            self.buy(size=position_size, sl=sl)
            print(f"🚀 MOON DEV LONG ENTRY: {price:.2f} | LVN: {sl:.2f} | Size: {position_size} ✨")

    def enter_short(self, price):
        sl = self.low_volume_node[-1]
        risk = sl - price
        if risk <= 0: return
        
        position_size = self.calculate_position_size(risk)
        if position_size > 0:
            self.sell(size=position_size, sl=sl)
            print(f"🌪️ MOON DEV SHORT ENTRY: {price:.2f} | LVN: {sl:.2f} | Size: {position_size} ✨")

    def calculate_position_size(self, risk):
        risk_amount = self.equity * self.risk_per_trade
        size = risk_amount / risk
        return int(round(size))

    def update_trailing_stops(self):
        if self.position.is_long:
            new_sl = self.lower_band[-1]
            if new_sl > self.position.sl:
                self.position.sl = new_sl
                print(f"🔼 LONG TRAILING STOP: {new_sl:.2f} 🌙")
        
        elif self.position.is_short:
            new_sl = self.upper_band[-1]