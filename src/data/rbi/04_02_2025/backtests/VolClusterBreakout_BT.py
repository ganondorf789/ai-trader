```python
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
from backtesting import Order

# Data handling
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class VolClusterBreakout(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    bb_period = 20
    bb_std = 2
    atr_period = 14
    swing_period = 20
    max_bars_held = 3

    def init(self):
        # 🌙 Bollinger Bands Calculation
        def calc_bb_upper(close):
            upper, _, _ = talib.BBANDS(close, timeperiod=self.bb_period, 
                                      nbdevup=self.bb_std, nbdevdn=self.bb_std)
            return upper
        self.bb_upper = self.I(calc_bb_upper, self.data.Close)

        def calc_bb_lower(close):
            _, _, lower = talib.BBANDS(close, timeperiod=self.bb_period,
                                     nbdevup=self.bb_std, nbdevdn=self.bb_std)
            return lower
        self.bb_lower = self.I(calc_bb_lower, self.data.Close)

        # 🌌 BB Width Percentile
        def bb_width_percentile(close):
            upper, middle, lower = talib.BBANDS(close, self.bb_period, 
                                              self.bb_std, self.bb_std)
            bb_width = (upper - lower) / middle
            return pd.Series(bb_width).rolling(100).rank(pct=True).values * 100
        self.bb_width_pct = self.I(bb_width_percentile, self.data.Close)

        # 🌠 Volatility Indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                        self.atr_period)
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period)

        self.bars_held = 0

    def next(self):
        # 🌙 Moon Dev Status Update
        if len(self.data) % 100 == 0:
            print(f"🌙 Moon Pulse: Bar {len(self.data)} | Price: {self.data.Close[-1]:.2f} | BB%: {self.bb_width_pct[-1]:.1f}")

        if self.position:
            self.bars_held += 1
            if self.bars_held >= self.max_bars_held:
                print(f"⏳ Moon Time Exit: Closing after {self.max_bars_held} bars")
                self.position.close()
            return

        # 🌌 Strategy Conditions
        bb_pct = self.bb_width_pct[-1]
        close_price = self.data.Close[-1]
        atr_value = self.atr[-1]

        if not (5 < bb_pct < 15) or not atr_value:
            return

        long_trigger = close_price > self.bb_upper[-1]
        short_trigger = close_price < self.bb_lower[-1]

        # 🌕 Long Entry Logic
        if long_trigger and self.swing_high[-1] >= close_price * 0.98:
            risk = 1.5 * atr_value
            position_size = self.calculate_position_size(risk)
            if position_size > 0:
                print(f"🚀🌕 MOON BLASTOFF LONG: {position_size} units")
                self.buy(size=position_size, 
                        sl=close_price - risk,
                        tp=self.swing_high[-1])

        # 🌑 Short Entry Logic
        elif short_trigger and self.swing_low[-1] <= close_price * 1.02:
            risk = 1.5 * atr