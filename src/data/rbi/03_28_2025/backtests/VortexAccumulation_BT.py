```python
# 🌙 Moon Dev's Vortex Accumulation Backtest 🌙
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib
import pandas_ta as ta

# Data preparation
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format data columns 🌟
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VortexAccumulation(Strategy):
    vi_period = 14
    cmf_period = 14
    risk_percent = 0.01  # 1% risk per trade
    swing_lookback = 20
    
    def init(self):
        # 🌀 Vortex Indicator Calculation
        vi_plus, vi_minus = ta.vortex(
            self.data.High, self.data.Low, self.data.Close, length=self.vi_period
        )
        self.vi_plus = self.I(lambda: vi_plus, name='VI+')
        self.vi_minus = self.I(lambda: vi_minus, name='VI-')

        # 💰 Chaikin Money Flow
        cmf = ta.cmf(
            self.data.High, self.data.Low, self.data.Close, self.data.Volume, length=self.cmf_period
        )
        self.cmf = self.I(lambda: cmf, name='CMF')

        # 🔍 Swing High/Low for Stops
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_lookback, name='Swing_Low')
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_lookback, name='Swing_High')

    def next(self):
        # Wait for sufficient data 🌙
        if len(self.data) < max(self.vi_period, self.swing_lookback) + 1:
            return

        # Get indicator values ✨
        vi_plus_c = self.vi_plus[-1]
        vi_plus_p = self.vi_plus[-2]
        vi_minus_c = self.vi_minus[-1]
        vi_minus_p = self.vi_minus[-2]
        
        cmf_c = self.cmf[-1]
        cmf_p = self.cmf[-2]
        price_c = self.data.Close[-1]
        price_p = self.data.Close[-2]

        # Long Entry: VI+ crosses VI+ & CMF Bull Divergence 🚀
        if not self.position:
            if crossover(vi_plus_c, vi_minus_c):
                if cmf_c > cmf_p and price_c <= price_p:
                    sl = self.swing_low[-1]
                    if sl and sl < price_c:
                        risk_amount = self.equity * self.risk_percent
                        risk_per_share = price_c - sl
                        size = int(round(risk_amount / risk_per_share))
                        if size > 0:
                            print(f"🌙✨ LONG! VI+↑VI- | CMF↑ @ {price_c:.2f} | Size: {size} 🚀")
                            self.buy(size=size, sl=sl)

            # Short Entry: VI- crosses VI+ & CMF Bear Divergence 💥
            elif crossover(vi_minus_c, vi_plus_c):
                if cmf_c < cmf_p and price_c >= price_p:
                    sl = self.swing_high[-1]
                    if sl and sl > price_c:
                        risk_amount = self.equity * self.risk_percent
                        risk_per_share = sl - price_c
                        size = int(round(risk_amount / risk_per_share))
                        if size > 0:
                            print(f"🌙✨ SHORT! VI-↑VI+ | CMF↓ @ {price_c:.2f} | Size: {size} 💥")
                            self.sell(size=size, sl=sl)

        # Exit Conditions 📉📈
        if self.position.is_long:
            if crossover(vi_minus_c, vi_plus_c) or cmf_c < 0:
                print(f"🌙 Closing LONG | VI-↑VI+ {vi_minus_c:.2f