Here's the fixed code with proper Moon Dev themed debug prints and all indicator packages properly implemented without any backtesting.lib usage:

```python
import pandas as pd
import pandas_ta as ta
import talib
from backtesting import Backtest, Strategy

# Load and preprocess data
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VortexContraction(Strategy):
    bb_period = 20
    bb_dev = 2
    vi_period = 14
    bb_width_ma_period = 20
    risk_per_trade = 0.01
    swing_period = 14
    take_profit_multiplier = 0.1
    time_exit_bars = 20

    def init(self):
        # Vortex Indicator using pandas_ta
        def calculate_vortex(high, low, close, period):
            df = ta.vortex(high=high, low=low, close=close, length=period)
            return df[f'VIp_{period}'].values, df[f'VIm_{period}'].values
        
        self.vi_plus, self.vi_minus = self.I(
            calculate_vortex, self.data.High, self.data.Low, self.data.Close, self.vi_period
        )

        # Bollinger Bands using talib
        self.upper, self.middle, self.lower = self.I(
            talib.BBANDS, self.data.Close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_dev,
            nbdevdn=self.bb_dev,
            matype=0
        )

        # Bollinger Width calculations
        self.bb_width = self.I(lambda u, l, m: (u - l) / m, self.upper, self.lower, self.middle)
        self.bb_width_ma = self.I(talib.SMA, self.bb_width, timeperiod=self.bb_width_ma_period)

        # Swing calculations using talib
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period)

        print("🌌✨ Moon Dev Indicators Initialized: Vortex + Bollinger Bands System Activated! ✨🌙")

    def next(self):
        if not self.position:
            # Common conditions
            bb_contraction = self.bb_width[-1] < self.bb_width_ma[-1]
            in_bands = self.lower[-1] < self.data.Close[-1] < self.upper[-1]

            # Long entry logic - using array indexing for crossover detection
            if (bb_contraction and in_bands and
                self.vi_plus[-1] > self.vi_minus[-1] and
                self.vi_plus[-2] <= self.vi_minus[-2]):
                
                sl_price = self.swing_low[-1]
                risk_amount = self.risk_per_trade * self.equity
                risk_per_share = self.data.Close[-1] - sl_price
                
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    print(f"🌙✨ BULLISH VORTEX CROSS detected! Entering LONG at {self.data.Close[-1]} ✨🚀")
                    print(f"🌕 Stop Loss set at {sl_price} | Position Size: {size} shares")
                    self.buy(size=size, sl=sl_price)

            # Short entry logic - using array indexing for crossover detection
            elif (bb_contraction and in_bands and
                  self.vi_minus[-1] > self.vi_plus[-1] and
                  self.vi_minus[-2] <= self.vi_plus[-2]):
                  
                sl_price = self.swing_high[-1]
                risk_amount = self.risk_per_trade * self.equity
                risk_per_share = sl_price - self.data.Close[-1]
                
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share