```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib
import numpy as np

# Data preprocessing
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class FibroVolume(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    ema_period = 50
    swing_window = 20
    rsi_period = 14
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    
    def init(self):
        # Core indicators 🌙
        self.ema = self.I(talib.EMA, self.data.Close, self.ema_period, name='EMA')
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_window, name='SWING_HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_window, name='SWING_LOW')
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period, name='RSI')
        self.macd_line, self.macd_signal, _ = self.I(talib.MACD, self.data.Close, 
                                                    self.macd_fast, self.macd_slow, self.macd_signal,
                                                    name='MACD')
        self.vol_sma = self.I(talib.SMA, self.data.Volume, 20, name='VOL_SMA')
        
        # Track trade parameters 🌌
        self.trade_fib_level = None
        self.entry_rsi = None

    def next(self):
        if self.position:
            self.manage_exit()
        else:
            self.check_entry()

    def check_entry(self):
        # Trend detection ✨
        current_close = self.data.Close[-1]
        trend = 'bullish' if current_close > self.ema[-1] else 'bearish'

        # Fibonacci calculations 🌗
        swing_range = self.swing_high[-1] - self.swing_low[-1]
        fib_levels = {
            '38.2%': self.swing_high[-1] - swing_range * 0.382,
            '50%': self.swing_high[-1] - swing_range * 0.5,
            '61.8%': self.swing_high[-1] - swing_range * 0.618
        } if trend == 'bullish' else {
            '38.2%': self.swing_low[-1] + swing_range * 0.382,
            '50%': self.swing_low[-1] + swing_range * 0.5,
            '61.8%': self.swing_low[-1] + swing_range * 0.618
        }

        # Fib level check with 0.5% tolerance 🎯
        for level, price in fib_levels.items():
            if abs(current_close - price) < current_close * 0.005:
                # Volume confirmation 📉
                if self.data.Volume[-1] < self.vol_sma[-1]:
                    print(f"🌙✨ {trend.upper()} FIBROVOLUME TRIGGER @ {level} ({price:.2f})! 🚀")
                    self.execute_trade(trend, current_close, price, fib_levels)
                    break

    def execute_trade(self, trend, entry_price, fib_price, fib_levels):
        # Risk management 🔒
        sl_price = min(fib_levels.values()) * 0.995 if trend == 'bullish' else max(fib_levels.values()) * 1.005
        risk_per_share = abs(entry_price - sl_price)
        position_size = int(round((self.equity * self.risk_percent) / risk_per_share))

        # Trade execution 🌙
        if trend == 'bullish':
            self.buy(size=position_size, sl=sl_price, tag='LONG')
            print(f"🚀🌙 BLASTOFF LONG: {position_size} units | Entry: {entry_price:.2f} | SL: {sl_price:.2f}")
        else: