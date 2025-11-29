I'll fix the code by removing all backtesting.lib imports and replacing crossover/crossunder functions with manual implementations. Here's the corrected version with Moon Dev themed debug prints:

```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

# Data loading and preprocessing
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure required columns
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Create dummy order_book_imbalance if missing
if 'order_book_imbalance' not in data.columns:
    data['order_book_imbalance'] = np.random.uniform(0, 100, len(data))

# Precompute daily VWAP
data['date'] = data.index.date
typical_price = (data['High'] + data['Low'] + data['Close']) / 3
data['cum_typical'] = data.groupby('date').apply(lambda x: (typical_price.loc[x.index] * x['Volume']).cumsum())
data['cum_vol'] = data.groupby('date')['Volume'].cumsum()
data['vwap'] = data['cum_typical'] / data['cum_vol']

# Precompute opening range
opening_range = data.groupby('date').head(2)
opening_high = opening_range.groupby('date')['High'].max().rename('opening_high')
opening_low = opening_range.groupby('date')['Low'].min().rename('opening_low')
data = data.join(opening_high, on='date')
data = data.join(opening_low, on='date')

data = data.drop(columns=['date', 'cum_typical', 'cum_vol'])

class VolatilityAnchoredBreakout(Strategy):
    vwap_multiplier = 1.5
    atr_period = 14
    volume_median_period = 15
    risk_pct = 0.01
    max_positions = 3
    daily_loss_limit = 0.03
    atr_stop = 0.75
    atr_take = 1.5
    
    def init(self):
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, 
                         self.data.Close, timeperiod=self.atr_period)
        self.volume_median = self.I(lambda x: x.rolling(self.volume_median_period).median(), 
                                  self.data.Volume)
        
        self.daily_pnl = {}
        self.current_date = None
        self.trading_allowed = True
        self.prev_upper = None
        self.prev_lower = None

    def next(self):
        # Moon Dev daily P&L tracking 🌙
        current_date = self.data.index[-1].date()
        if current_date != self.current_date:
            self.current_date = current_date
            self.daily_pnl[current_date] = 0
            self.trading_allowed = True
            print(f"🌙 New Moon Cycle Starting: {current_date}")

        if not self.trading_allowed:
            return

        # Moon Dev time filter 🌓
        current_time = self.data.index[-1].time()
        if current_time < pd.Timestamp('00:15').time() or \
           current_time > pd.Timestamp('23:45').time():
            return

        # Moon Dev position management 🚀
        if len(self.positions) >= self.max_positions:
            return

        # Calculate current indicators
        vwap = self.data.vwap[-1]
        atr = self.atr[-1]
        upper_band = vwap + atr * self.vwap_multiplier
        lower_band = vwap - atr * self.vwap_multiplier
        vol_median = self.volume_median[-1]
        obi = self.data.order_book_imbalance[-1]

        # Moon Dev entry signals 🌙✨
        long_signal = False
        short_signal = False
        
        if len(self.data.Close) > 1:
            # Bullish crossover replacement
            if self.data.Close[-2] < upper_band and self.data.Close[-1] > upper_band:
                long