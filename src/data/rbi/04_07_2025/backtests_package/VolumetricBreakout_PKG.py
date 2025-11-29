Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations:

```python
# 🌙 Moon Dev's VolumetricBreakout Backtest 🌙
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

class VolumetricBreakout(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # 🌌 Calculate Bollinger Bands with Width using TA-Lib
        def bbands_with_width(close):
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            width = upper - lower
            return upper, middle, lower, width
        
        self.upper, self.middle, self.lower, self.bb_width = self.I(
            bbands_with_width, self.data.Close, name=['Upper', 'Middle', 'Lower', 'Width']
        )
        
        # 📈 Volume MA indicator
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=50, name='Volume MA')
        
        print("🌙 Moon Dev Indicators Initialized! 🚀")

    def next(self):
        # 🛑 Skip early bars without indicator data
        if len(self.data) < 100 or np.isnan(self.bb_width[-1]):
            return
        
        # 🌐 Current market conditions
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        volume_ma = self.volume_ma[-1]
        bb_width = self.bb_width[-1]
        
        # 📉 Calculate Bollinger contraction (20th percentile of last 100 widths)
        recent_widths = self.bb_width[-100:]
        contraction_threshold = np.percentile(recent_widths, 20)
        contraction = bb_width <= contraction_threshold
        
        # 💹 Volume surge condition
        volume_surge = current_volume > volume_ma
        
        # 🚀 Entry Logic
        if not self.position:
            if contraction and volume_surge:
                # Long Entry
                if current_close > self.upper[-1]:
                    stop_price = self.lower[-1]
                    risk_per_share = current_close - stop_price
                    if risk_per_share > 0:
                        position_size = int(round((self.equity * self.risk_percent) / risk_per_share))
                        if position_size > 0:
                            self.buy(size=position_size)
                            print(f"🌙 MOON DEV LONG SIGNAL! 🚀 Size: {position_size} | Price: {current_close:.2f} ✨")
                
                # Short Entry    
                elif current_close < self.lower[-1]:
                    stop_price = self.upper[-1]
                    risk_per_share = stop_price - current_close
                    if risk_per_share > 0:
                        position_size = int(round((self.equity * self.risk_percent) / risk_per_share))
                        if position_size > 0:
                            self.sell(size=position_size)
                            print(f"🌙 MOON DEV SHORT SIGNAL! 🌑 Size: {position_size} | Price: {current_close:.2f} ✨")
        
        # 💰 Exit Logic
        else:
            if self.position.is_long and current_close <= self.middle[-1]:
                self.position.close()
                print(f"🌙 MOON DEV EXIT LONG! ✅ Price: {current_close:.2f} 🏁")
            
            elif self.position.is_short and current_close >= self.middle[-1]:
                self.position.close()
                print(f"🌙 MOON DEV EXIT SHORT! ✅ Price: {current_close:.2f} 🏁")

# 🧹 Data Cleaning
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# 🗺️ Column Mapping
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# ⏰ Datetime Index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# 🚀 Launch Backtest
print("🌙 MOON DEV BACKTEST INITIATED! 🚀\n")
bt = Backtest(data, Volum