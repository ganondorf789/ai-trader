```python
# 🌙 Moon Dev's VolatileBreakout Backtest Script 🚀

import pandas as pd
import talib
from backtesting import Strategy, Backtest
from backtesting.lib import crossover, crossunder

# ========================
# DATA PREPARATION ✨
# ========================
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean and prepare data 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map columns to backtesting.py format 🗺️
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# ========================
# STRATEGY IMPLEMENTATION 🚀
# ========================
class VolatileBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    atr_period = 20
    vol_lookback = 30
    swing_period = 20
    
    def init(self):
        # 🌙 Core Indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.atr_high = self.I(talib.MAX, self.atr, timeperiod=90)  # 90-day ATR high
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_lookback)
        
        # 🌙 Swing Points Calculation
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period)
        
        # 🌙 Track trade parameters
        self.entry_price = None
        self.extreme_price = None

    def next(self):
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        
        # 🌙 Initialize trade parameters
        if not self.position:
            # Long Entry Conditions 🌙
            long_trigger = current_close > self.swing_high[-1]
            volatility_ok = self.atr[-1] > self.atr_high[-1]
            volume_ok = self.data.Volume[-1] > self.volume_ma[-1]
            
            if long_trigger and volatility_ok and volume_ok:
                # Calculate position size 🌙
                stop_loss = self.swing_low[-1]
                risk_amount = self.equity * self.risk_per_trade
                risk_per_unit = self.data.Close[-1] - stop_loss
                
                if risk_per_unit > 0:
                    position_size = int(round(risk_amount / risk_per_unit))
                    self.buy(size=position_size, sl=stop_loss)
                    print(f"🌙 LONG ENTRY! Size: {position_size} @ {self.data.Close[-1]:.2f} | SL: {stop_loss:.2f} 🚀")
                    self.entry_price = self.data.Close[-1]
                    self.extreme_price = current_high  # Track initial extreme

            # Short Entry Conditions 🌙 (mirror logic)
            short_trigger = current_close < self.swing_low[-1]
            if short_trigger and volatility_ok and volume_ok:
                stop_loss = self.swing_high[-1]
                risk_amount = self.equity * self.risk_per_trade
                risk_per_unit = stop_loss - self.data.Close[-1]
                
                if risk_per_unit > 0:
                    position_size = int(round(risk_amount / risk_per_unit))
                    self.sell(size=position_size, sl=stop_loss)
                    print(f"🌙 SHORT ENTRY! Size: {position_size} @ {self.data.Close[-1]:.2f} | SL: {stop_loss:.2f} 💥")
                    self.entry_price = self.data.Close[-1]
                    self.extreme_price = current_low

        # 🌙 Manage open positions
        else:
            # Update extreme price 🌙
            if self.position.is_long:
                self.extreme_price = max(self.extreme_price, current_high)
                threshold = self.extreme_price - 0.5*(self.extreme_price - self.entry_price)
                if