Here's the fixed code with all backtesting.lib references removed and replaced with proper Moon Dev themed implementations:

```python
# 🌙 Moon Dev Backtest Implementation for VoltaicRebound Strategy
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# Data Preparation 🌍
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and prepare data columns 🧹
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

class VoltaicRebound(Strategy):
    def init(self):
        # 🌗 Moon Dev Indicators Setup
        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, 20, 2, 2, 0
        )
        
        # Chaikin Money Flow with SMA
        cmf_values = ta.cmf(
            high=self.data.High, low=self.data.Low,
            close=self.data.Close, volume=self.data.Volume, length=20
        )
        self.cmf = self.I(lambda: cmf_values, name='CMF_20')
        self.cmf_sma = self.I(talib.SMA, self.cmf, 10)
        
        # Donchian Channel System 🌪️
        self.donchian_high = self.I(talib.MAX, self.data.High, 20)
        self.donchian_low = self.I(talib.MIN, self.data.Low, 20)
        self.donchian_mid = self.I(
            lambda h, l: (h + l)/2, self.donchian_high, self.donchian_low
        )
        
        # Volatility Filter Calculations
        self.donchian_width = self.I(
            lambda h, l: h - l, self.donchian_high, self.donchian_low
        )
        self.width_min = self.I(talib.MIN, self.donchian_width, 100)
        self.width_max = self.I(talib.MAX, self.donchian_width, 100)
        
        print("🌙✨ Moon Dev Trading Engines Activated! 🚀")

    def next(self):
        # 🌑 Skip early bars without indicator data
        if len(self.data) < 100 or \
            any(pd.isna([self.bb_lower[-1], self.cmf[-1], self.cmf_sma[-1],
                        self.donchian_mid[-1], self.width_min[-1], self.width_max[-1]])):
            return

        # 🌗 Moon Dev Volatility Filter
        current_width = self.donchian_width[-1]
        p20_level = self.width_min[-1] + 0.2*(self.width_max[-1] - self.width_min[-1])
        if current_width <= p20_level:
            print(f"🌑 Moon Dev Alert: Volatility Contraction ({current_width:.2f} ≤ {p20_level:.2f})")
            return

        # 🚀 Long Entry Logic
        if not self.position:
            # Bollinger Touch + CMF Cross Conditions
            bb_touch = self.data.Low[-1] <= self.bb_lower[-1]
            cmf_cross = (self.cmf[-2] < self.cmf_sma[-2]) and (self.cmf[-1] > self.cmf_sma[-1])
            
            if bb_touch and cmf_cross:
                # 🌙 Risk Calculation
                risk_amount = self.equity * 0.01  # 1% risk
                entry_price = self.data.Open[-1]
                stop_price = self.bb_lower[-1]
                risk_per_share = entry_price - stop_price
                
                if risk_per_share <= 0:
                    print("🌑 Moon Dev Warning: Negative Risk Detected!")
                    return
                
                position_size = int(round(risk_amount / risk_per_share))
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_price,
                            tag="🌙 VoltaicRebound Entry")
                    print