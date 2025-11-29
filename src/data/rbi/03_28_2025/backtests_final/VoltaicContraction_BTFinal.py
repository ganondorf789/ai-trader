Here's the complete fixed backtest code with Moon Dev themed debug prints and proper position sizing implementation:

```python
# 🌙 Moon Dev's VoltaicContraction Backtest Implementation ✨
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import numpy as np

# Data preparation moon magic ✨
def load_data(path):
    data = pd.read_csv(path)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Drop unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    # Proper column mapping
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    return data

data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = load_data(data_path)

class VoltaicContraction(Strategy):
    # Strategy parameters 🌗
    bb_period = 20
    bb_dev = 2
    adx_period = 14
    rsi_period = 14
    atr_period = 14
    bandwidth_lookback = 960  # 10 days in 15m intervals
    risk_pct = 0.01
    adx_threshold = 20
    rsi_exit = 70
    atr_multiplier = 1.5

    def init(self):
        # Indicator calculation magic ✨
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Bollinger Bands components 🌈
        def bb_upper(close): return talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)[0]
        def bb_middle(close): return talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)[1]
        def bb_lower(close): return talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)[2]
        
        self.bb_upper = self.I(bb_upper, close)
        self.bb_middle = self.I(bb_middle, close)
        self.bb_lower = self.I(bb_lower, close)
        
        # Volatility contraction metrics 🌗
        self.bandwidth = self.I(lambda: (self.bb_upper - self.bb_lower) / self.bb_middle)
        self.min_bandwidth = self.I(talib.MIN, self.bandwidth, self.bandwidth_lookback)
        
        # Trend strength indicators 🌊
        self.adx = self.I(talib.ADX, high, low, close, self.adx_period)
        self.rsi = self.I(talib.RSI, close, self.rsi_period)
        self.atr = self.I(talib.ATR, high, low, close, self.atr_period)
        
        # Risk management state 🛡️
        self.consecutive_losses = 0
        self.entry_bandwidth = None

    def next(self):
        # Moon Dev debug monitoring 🌙
        print(f"🌙 Bar {len(self.data)} | Close: {self.data.Close[-1]:.2f} | "
              f"BW: {self.bandwidth[-1]:.4f} | ADX: {self.adx[-1]:.1f} | "
              f"RSI: {self.rsi[-1]:.1f} | ATR: {self.atr[-1]:.2f}")

        if not self.position and self.consecutive_losses < 3:
            # Entry conditions check 🌌
            price_in_bands = (self.data.Close[-1] > self.bb_lower[-1]) and \
                             (self.data.Close[-1] < self.bb_upper[-1])
            volatility_contraction = self.bandwidth[-1] <= self.min_bandwidth[-1]
            weak_trend = self.adx[-1] < self.adx_threshold

            if all([volatility_contraction, weak_trend, price_in_bands]):
                # Position sizing calculation 🧮
                risk_amount = self.equity * self.risk_pct
                stop_loss_dist = self.atr[-1] * self.atr_mult