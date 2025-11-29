Here's the fully debugged and Moon Dev enhanced version of your LiquidationCluster strategy:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Strategy, Backtest

class LiquidationCluster(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_dev = 2
    swing_period = 20
    bb_width_threshold = 0.05  # Volatility contraction threshold
    exit_multiplier = 2.0       # BB width expansion multiplier for exit
    volume_ma_period = 20
    max_bars_held = 5          # Time-based exit threshold
    
    def init(self):
        # Calculate indicators using TA-Lib through self.I()
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(talib.BBANDS, 
            self.data.Close, self.bb_period, self.bb_dev, self.bb_dev, 0)
        self.bb_width = (self.bb_upper - self.bb_lower) / self.bb_middle
        self.I(self.bb_width, name='BB_WIDTH')
        
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period)
        self.volume_ma = self.I(talib.SMA, self.data.Volume, self.volume_ma_period)
        
        print("🌙✨ Moon Dev Strategy Activated! Ready to hunt liquidations! 🚀")
        print("✨ All indicators powered by pure TA-Lib - No backtesting.lib used! 🌙")

    def next(self):
        price = self.data.Close[-1]
        
        # Moon Dev progress tracking
        if len(self.data) % 100 == 0:
            print(f"🌙 Moon Dev Progress: Bar {len(self.data)} | Price: {price:.2f} | BB Width: {self.bb_width[-1]:.4f}")

        # Exit conditions
        if self.position:
            # Volatility expansion exit
            if self.bb_width[-1] > (self.bb_width[self.position.entry_bar] * self.exit_multiplier):
                print(f"🚀✨ Moon Dev Exit: Volatility Explosion! Closing position at {price:.2f}")
                self.position.close()
            
            # Time-based exit
            elif (len(self.data) - self.position.entry_bar) >= self.max_bars_held:
                print(f"⏳🌙 Moon Dev Exit: Time Limit Reached! Closing after {self.max_bars_held} bars")
                self.position.close()

        # Entry conditions
        else:
            if self.bb_width[-1] < self.bb_width_threshold:
                long_trigger = (price > self.swing_high[-1]) and (self.data.Volume[-1] > self.volume_ma[-1])
                short_trigger = (price < self.swing_low[-1]) and (self.data.Volume[-1] > self.volume_ma[-1])

                if long_trigger or short_trigger:
                    # Risk management calculations
                    risk_percent = 0.01  # 1% risk per trade
                    stop_pct = 0.01      # 1% stop from entry
                    
                    if long_trigger:
                        stop_price = self.swing_low[-1]
                        direction = "LONG"
                    else:
                        stop_price = self.swing_high[-1]
                        direction = "SHORT"
                    
                    risk_amount = self.equity * risk_percent
                    risk_distance = abs(price - stop_price)
                    position_size = risk_amount / risk_distance
                    position_size = int(round(position_size))  # Ensure whole number units
                    
                    # Execute trade with Moon Dev flair
                    if position_size > 0:
                        print(f"🌙🚀 MOON SHOT! {direction} Entry at {price:.2f} | Size: {position_size:,}")
                        if long_trigger:
                            self.buy(size=position_size, sl=stop_price)
                        else:
                            self.sell(size=position_size, sl=stop_price)

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = [col.strip().lower() for col in data.columns]
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',