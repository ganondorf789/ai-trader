Here's the implementation of the LiquidationReversal strategy for backtesting.py:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Load and preprocess data
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class LiquidationReversal(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    swing_window = 20
    rsi_period = 14
    buffer_pct = 0.01  # 1% buffer for stop loss

    def init(self):
        # Resample to 4H for RSI calculation 🌙
        resampled_close = self.data.Close.resample('4H').last().ffill()
        self.rsi_4h = self.I(talib.RSI, resampled_close, self.rsi_period, name='RSI 4H')
        
        # Liquidation zone detection ✨
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_window, name='Swing High')
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_window, name='Swing Low')
        
        # Volume-weighted pivot calculation 🚀
        typical_price = (self.data.High + self.data.Low + self.data.Close) / 3
        vwap = typical_price * self.data.Volume
        self.vwap_sma = self.I(talib.SMA, vwap, self.swing_window, name='VWAP Pivot')

    def next(self):
        if len(self.data) < self.swing_window or len(self.rsi_4h) < 2:
            return

        # Current market conditions 🌙
        price = self.data.Close[-1]
        vwap_pivot = self.vwap_sma[-1]
        current_rsi = self.rsi_4h[-1]
        prev_rsi = self.rsi_4h[-2]

        if not self.position:
            # Long entry: RSI reversal from oversold 🚀
            if (cross_over(prev_rsi, current_rsi, 20) and
                price <= self.swing_low[-1] * (1 + self.buffer_pct)):
                self.enter_long(price, vwap_pivot)

            # Short entry: RSI reversal from overbought 🌑
            elif (cross_under(prev_rsi, current_rsi, 80) and
                  price >= self.swing_high[-1] * (1 - self.buffer_pct)):
                self.enter_short(price, vwap_pivot)

    def enter_long(self, entry_price, tp_price):
        sl_price = self.swing_low[-1] * (1 - self.buffer_pct)
        risk = entry_price - sl_price
        if risk <= 0: return

        position_size = self.calculate_position_size(risk)
        if position_size > 0:
            self.buy(size=position_size, sl=sl_price, tp=tp_price)
            print(f"🌙✨ BULLISH REVERSAL! Buying {position_size} @ {entry_price:.2f}")
            print(f"   SL: {sl_price:.2f} | TP: {tp_price:.2f} 🚀\n")

    def enter_short(self, entry_price, tp_price):
        sl_price = self.swing_high[-1] * (1 + self.buffer_pct)
        risk = sl_price - entry_price
        if risk <= 0: return

        position_size = self.calculate_position_size(risk)
        if position_size > 0:
            self.sell(size=position_size, sl=sl_price, tp=tp_price)
            print(f"🌙✨ BEARISH REVERSAL! Selling {position_size} @ {entry_price:.2f}")
            print(f"   SL: {sl_price:.2f} | TP: {tp_price:.2f} 🌑\n")

    def calculate_position_size(self, risk_per_share):
        risk_amount = self.risk_pct * self.equity
        position_size = risk