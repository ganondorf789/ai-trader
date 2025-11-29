Here's the fixed code with all backtesting.lib references removed and Moon Dev themed debug prints added:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class VolumetricCloud(Strategy):
    def init(self):
        # 🌙 Calculate Volume Indicators
        self.ema5_vol = self.I(talib.EMA, self.data.Volume, timeperiod=5, name='EMA5 Vol')
        self.ema20_vol = self.I(talib.EMA, self.data.Volume, timeperiod=20, name='EMA20 Vol')
        
        # ✨ Volume Difference and STDDEV
        def calculate_diff(data):
            return talib.EMA(data.Volume, timeperiod=5) - talib.EMA(data.Volume, timeperiod=20)
        self.diff_vol = self.I(calculate_diff, name='DIFF Vol')
        self.std_diff = self.I(talib.STDDEV, self.diff_vol, timeperiod=20, name='STDDEV Diff')

        # 🚀 Ichimoku Cloud Components
        def tenkan_sen(data):
            high = talib.MAX(data.High, timeperiod=9)
            low = talib.MIN(data.Low, timeperiod=9)
            return (high + low) / 2
        self.tenkan = self.I(tenkan_sen, name='Tenkan')

        def kijun_sen(data):
            high = talib.MAX(data.High, timeperiod=26)
            low = talib.MIN(data.Low, timeperiod=26)
            return (high + low) / 2
        self.kijun = self.I(kijun_sen, name='Kijun')

        def senkou_a(data):
            return (self.tenkan + self.kijun) / 2
        self.senkou_a = self.I(senkou_a, name='Senkou A')

        def senkou_b(data):
            high = talib.MAX(data.High, timeperiod=52)
            low = talib.MIN(data.Low, timeperiod=52)
            return (high + low) / 2
        self.senkou_b = self.I(senkou_b, name='Senkou B')

    def next(self):
        current_idx = len(self.data) - 1
        
        # 🌙 Check minimum data requirements
        if current_idx < 26:
            print("🌑 MOON DEV: Waiting for sufficient data...")
            return

        # ✨ Get cloud values from 26 periods ago
        cloud_a = self.senkou_a[current_idx-26]
        cloud_b = self.senkou_b[current_idx-26]
        current_close = self.data.Close[current_idx]

        # 🚀 Entry Conditions
        volume_surge = (self.ema5_vol[current_idx] > 
                       (self.ema20_vol[current_idx] + 2 * self.std_diff[current_idx]))
        price_above_cloud = current_close > max(cloud_a, cloud_b)

        # 🌟 Long Entry Logic
        if not self.position and volume_surge and price_above_cloud:
            # Risk Management
            risk_percent = 0.01  # 1% of equity
            risk_amount = self.equity * risk_percent
            position_size = risk_amount / current_close
            position_size = int(round(position_size))
            
            if position_size > 0:
                self.buy(size=position_size, trailing_sl=0.05)
                print(f"🌕✨🚀 MOON DEV ALERT: LONG ENTRY @ {current_close:.2f}")
                print(f"📈 Size: {position_size} | Equity: {self.equity:.2f}")
                print(f"📊 Volume Surge: {self.ema5_vol[current_idx]:.2f} > {self.ema20_vol[current_idx]:.2f} + 2*STD")
                print(f"☁️  Price Above Cloud: {current_close:.2f} > Cloud Top {max(cloud_a, cloud_b):.2f}")

        # 💫 Exit handled automatically by trailing stop
        elif self.position and self.position.is_long:
            print(f"🌓 MOON DEV: Tracking long position with trailing stop...")

# 🌙 Data Preparation
print("🌑 MOON DEV: Loading celestial market data...")
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.loc[:, ~data.columns.str.contains('unnamed')]

# Format columns