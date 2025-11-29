Here's the fixed code with all backtesting.lib imports removed and replaced with proper Moon Dev implementations:

```python
# 🌙 Moon Dev's Volatility Breakout Backtest 🌙
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# ========================
# DATA PREPARATION
# ========================
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and prepare data 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# ========================
# STRATEGY IMPLEMENTATION
# ========================
class VolatilityBreakout(Strategy):
    ema_period_fast = 20
    ema_period_slow = 50
    atr_period = 14
    atr_sma_period = 20
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # 🌙 Trend Indicators
        self.ema_fast = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period_fast)
        self.ema_slow = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period_slow)
        
        # 🌈 Volatility Indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.atr_sma = self.I(talib.SMA, self.atr, timeperiod=self.atr_sma_period)
        
        print("🌙 Strategy Initialized with Moon Power! ✨")

    def next(self):
        # Wait for sufficient data 🌌
        if len(self.data) < max(self.ema_period_slow, self.atr_sma_period) + 3:
            return

        # 🌙 Trend Confirmation Check (replaced crossover with manual check)
        ema_bullish = (self.ema_fast[-2] < self.ema_slow[-2]) and (self.ema_fast[-1] > self.ema_slow[-1])
        
        # 🌊 Volatility Check (current ATR < SMA of ATR)
        low_volatility = self.atr[-1] < self.atr_sma[-1]

        # 🚀 ENTRY LOGIC
        if not self.position and ema_bullish and low_volatility:
            # Calculate position size with Moon Precision 🌙
            risk_amount = self.equity * self.risk_per_trade
            atr_value = self.atr[-1]
            
            if atr_value > 0:
                position_size = risk_amount / atr_value
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.buy(size=position_size, tag="🌙 Moon Rocket Launch!")
                    print(f"🚀 LIFT OFF! | Size: {position_size} | Entry: {self.data.Close[-1]} | ATR: {atr_value:.2f}")

        # 🔥 EXIT LOGIC
        if self.position:
            # Update trailing stop with Moon Gravity 🌑
            current_high = self.data.High[-1]
            trail_stop = current_high - (2 * self.atr[-1])
            
            # Maintain highest high since entry 🌕
            if current_high > self.position.high:
                self.position.high = current_high
                print(f"✨ NEW ORBIT HIGH: {current_high:.2f}")

            # Check trailing stop exit
            if self.data.Close[-1] < trail_stop:
                self.sell(tag="🌙 Moon Landing (Trailing Stop)")
                print(f"🔻 LANDING SEQUENCE | Price: {self.data.Close[-1]} | Trail Stop: {trail_stop:.2f}")

            # Time-based exit (10 bars max) ⏳
            if len(self.data) - self.position.entry_bar >= 10:
                self.sell(tag="🌙 Moon Mission Timeout")
                print(f"⏰ MISSION DURATION EXIT | Bars Held: {len(self.data