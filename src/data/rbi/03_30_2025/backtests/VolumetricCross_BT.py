Here's the complete backtesting implementation for the VolumetricCross strategy:

```python
# 🌙 Moon Dev Backtest AI Implementation ✨
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib
import pandas as pd
import numpy as np

# ========== DATA PREPARATION ==========
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and format column names 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to proper column casing ✨
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert and set datetime index 🕒
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# ========== STRATEGY CLASS ==========
class VolumetricCross(Strategy):
    risk_percent = 0.01  # Risk 1% per trade 🌙
    stop_loss_pct = 0.01  # 1% stop loss ✨
    take_profit_pct = 0.02  # 2% take profit 🚀
    
    def init(self):
        # ===== INDICATOR CALCULATIONS =====
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=50)
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=200)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
        # Moon Dev Debug Setup 🌙
        print("✨ Initializing Moon Dev VolumetricCross Strategy ✨")
        print(f"🌙 Risk Parameters: {self.risk_percent*100}% Risk | 1:{self.take_profit_pct/self.stop_loss_pct} R/R")

    def next(self):
        # ===== ENTRY LOGIC =====
        if not self.position:
            # Check Golden Cross condition 🌙
            if crossover(self.ema50, self.ema200):
                # Verify volume spike ✨
                current_volume = self.data.Volume[-1]
                volume_avg = self.volume_sma[-1]
                
                if current_volume >= 2 * volume_avg:
                    # Calculate position sizing 🚀
                    entry_price = self.data.Close[-1]
                    stop_price = entry_price * (1 - self.stop_loss_pct)
                    tp_price = entry_price * (1 + self.take_profit_pct)
                    
                    # Risk-based position sizing 🌙
                    risk_amount = self.equity * self.risk_percent
                    risk_per_share = entry_price - stop_price
                    position_size = int(round(risk_amount / risk_per_share))
                    
                    # Execute trade with OCO orders ✨
                    self.buy(size=position_size,
                            sl=stop_price,
                            tp=tp_price)
                    
                    # Moon Dev Trade Alert 🚀
                    print(f"\n🌙✨🚀 MOON DEV ENTRY SIGNAL 🌙✨🚀"
                          f"\nEntry Price: {entry_price:.2f}"
                          f"\nPosition Size: {position_size} units"
                          f"\nStop Loss: {stop_price:.2f} (-{self.stop_loss_pct*100}%)"
                          f"\nTake Profit: {tp_price:.2f} (+{self.take_profit_pct*100}%)")

        # ===== EXIT LOGIC =====
        else:
            # RSI-based exit condition 🌙
            if self.rsi[-1] < 70 and self.rsi[-2] >= 70:
                self.position.close()
                print(f"\n🌑✨🚀 MOON DEV EXIT SIGNAL (RSI <70) 🌑✨🚀"
                      f"\nExit Price: {self.data.Close[-1]:.2f}")

# ========== BACKTEST EXECUTION ==========
if __name__ == "__main__":
    # Initialize backtest with 1M capital 🌙
    bt = Backtest(data, VolumetricCross, cash=1_000_000)
    
    # Run and print full statistics ✨
    stats = bt.run()
    print("\n🌙✨