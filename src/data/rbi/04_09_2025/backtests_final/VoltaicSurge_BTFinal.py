Here's the fully debugged and optimized backtest code for the Voltaic Surge strategy with all Moon Dev enhancements:

```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR VOLTAIC SURGE STRATEGY 🚀
# DEBUG COMPLETE - LUNAR CODE OPTIMIZED ✨

# Required imports
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

class VoltaicSurge(Strategy):
    # Strategy parameters 🌟
    ema_period = 200
    atr_period = 14
    atr_ma_period = 20
    stop_multiplier = 2
    risk_pct = 0.01  # 1% risk per trade
    max_exposure_pct = 0.05  # 5% max exposure
    
    def init(self):
        # 🌙 Moon Dev Indicators Setup
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                         timeperiod=self.atr_period)
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=self.atr_ma_period)
        self.highest_high = None  # For trailing stop calculation
        
        print("🌙✨ MOON DEV: Indicators initialized with celestial precision!")
        
    def next(self):
        # Wait for indicators to warm up 🌊
        if len(self.data.Close) < self.ema_period + 1:
            return
            
        current_close = self.data.Close[-1]
        current_ema = self.ema[-1]
        current_atr = self.atr[-1]
        current_atr_ma = self.atr_ma[-1]
        
        # 🌙✨ Entry Logic: Volatility Surge with Trend Confirmation
        if not self.position:
            # Check crossover conditions using proper array indexing
            atr_cross = (self.atr[-2] <= self.atr_ma[-2]) and (current_atr > current_atr_ma)
            
            if current_close > current_ema and atr_cross:
                # 🌙 Risk Management Calculations
                risk_amount = self.equity * self.risk_pct
                atr_value = current_atr
                price_per_unit = current_close
                
                # Calculate position size with multiple constraints
                position_size = risk_amount / (atr_value * self.stop_multiplier)
                max_size = (self.equity * self.max_exposure_pct) / price_per_unit
                position_size = min(position_size, max_size)
                
                # 🌙 CRITICAL FIX: Ensure whole number units
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    self.highest_high = self.data.High[-1]  # Initialize trailing stop
                    print(f"🌙🚀 MOON DEV ENTRY: Long {position_size} units at {current_close:.2f}")
                    print(f"   - Stop Distance: {atr_value * self.stop_multiplier:.2f} points")
        
        # 🌑 Exit Management for open positions
        else:
            # Update trailing stop levels
            self.highest_high = max(self.highest_high, self.data.High[-1])
            stop_price = self.highest_high - (current_atr * self.stop_multiplier)
            
            # Trailing stop exit
            if self.data.Low[-1] < stop_price:
                self.position.close()
                print(f"🌙🛑 MOON DEV TRAILING STOP: Exit at {self.data.Close[-1]:.2f}")
                print(f"   - Profit: {self.position.pl:.2f} ({self.position.pl_pct:.2%})")
            
            # Emergency trend failure exit
            elif current_close < current_ema:
                self.position.close()
                print(f"🌙⚠️ EMERGENCY EXIT: Price closed below EMA at {current_close:.2f}")
                print(f"   - Profit: {self.position.pl:.2f} ({self.position.pl_pct:.2%})")

# 🌙 Data Preparation Ritual
print("\n🌙 INITIATING LUNAR DATA CLEANSING CEREMONY...")
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# 🧹 Data Cleansing Ceremony
data.columns = data.columns.str.strip