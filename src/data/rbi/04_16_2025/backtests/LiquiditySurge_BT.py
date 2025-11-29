```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION FOR LIQUIDITYSURGE STRATEGY

# Required imports
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import resample_apply

class LiquiditySurge(Strategy):
    # Strategy parameters
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌙 MOON DEV INDICATOR CALCULATION
        # Clean data columns handled externally
        
        # Calculate Keltner Channels components
        self.ema20 = self.I(talib.EMA, self.data.Close, timeperiod=20)
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.upper_kc = self.I(lambda ema, atr: ema + 1.5*atr, self.ema20, self.atr14)
        self.lower_kc = self.I(lambda ema, atr: ema - 1.5*atr, self.ema20, self.atr14)
        
        # Volume indicators
        self.volume_sma20 = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
        # Initialize trade tracking variables
        self.current_take_profit = None
        self.current_stop_loss = None
        self.entry_upper_kc = None
        self.entry_lower_kc = None

    def next(self):
        # 🌙 MOON DEV TRADE LOGIC
        price = self.data.Close[-1]
        upper = self.upper_kc[-1]
        lower = self.lower_kc[-1]
        volume = self.data.Volume[-1]
        vol_sma = self.volume_sma20[-1]
        
        if not self.position:
            # 🌟 LONG ENTRY CONDITION
            if price > upper and volume > 4*vol_sma:
                self._enter_long(price, upper, lower)
            
            # 🌪️ SHORT ENTRY CONDITION
            elif price < lower and volume > 4*vol_sma:
                self._enter_short(price, upper, lower)
        
        else:
            # 🎯 Check exit conditions
            if self.position.is_long:
                self._check_long_exits(price)
            elif self.position.is_short:
                self._check_short_exits(price)

    def _enter_long(self, price, upper, lower):
        risk_share = price - lower
        if risk_share <= 0:
            print("🌙🚨 INVALID LONG RISK")
            return
        
        size = int(round((self.equity * self.risk_percent) / risk_share))
        if size <= 0:
            print("🌙🚨 ZERO LONG SIZE")
            return
        
        self.current_take_profit = price + 1.5*(upper - lower)
        self.current_stop_loss = lower
        self.entry_upper_kc = upper
        
        self.buy(size=size)
        print(f"🚀🌙 LONG ENTRY | Price: {price:.2f} | Size: {size} | TP: {self.current_take_profit:.2f} | SL: {lower:.2f}")

    def _enter_short(self, price, upper, lower):
        risk_share = upper - price
        if risk_share <= 0:
            print("🌙🚨 INVALID SHORT RISK")
            return
        
        size = int(round((self.equity * self.risk_percent) / risk_share))
        if size <= 0:
            print("🌙🚨 ZERO SHORT SIZE")
            return
        
        self.current_take_profit = price - 1.5*(upper - lower)
        self.current_stop_loss = upper
        self.entry_lower_kc = lower
        
        self.sell(size=size)
        print(f"📉🌙 SHORT ENTRY | Price: {price:.2f} | Size: {size} | TP: {self.current_take_profit:.2f} | SL: {upper:.2f}")

    def _check_long_exits(self, price):
        # 🎯 Take Profit
        if self.data.High[-1] >= self.current_take_profit:
            self.position.close()
            print(f"🎯🌙 LONG TP HIT | Price: {price:.2f}")
        
        # 🛑 Stop Loss
        elif self.data.Low[-1] <= self.current_stop_loss:
            self.position.close()
            print(f"🛑