```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib as ta
import numpy as np

# Load and prepare data
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class DeltaSurgeReversal(Strategy):
    risk_pct = 0.01
    rsi_period = 14
    swing_window = 20
    volume_lookback = 50
    
    def init(self):
        # Core strategy indicators
        self.rsi = self.I(ta.RSI, self.data.Close, self.rsi_period)
        self.swing_high = self.I(ta.MAX, self.data.High, self.swing_window)
        self.swing_low = self.I(ta.MIN, self.data.Low, self.swing_window)
        self.volume_threshold = self.I(
            lambda x: x.rolling(self.volume_lookback).quantile(0.9),
            self.data.Volume
        )
        
    def next(self):
        price = self.data.Close[-1]
        
        if not self.position:
            # 🌙 Moon Dev Entry Logic
            if self._long_signal():
                self._enter_long()
                
            elif self._short_signal():
                self._enter_short()

    def _long_signal(self):
        return (self.rsi[-1] < 30 
                and self.data.Volume[-1] > self.volume_threshold[-1])
    
    def _short_signal(self):
        return (self.rsi[-1] > 70 
                and self.data.Volume[-1] > self.volume_threshold[-1])
    
    def _enter_long(self):
        swing_low = self.swing_low[-1]
        risk_amount = self.equity * self.risk_pct
        stop_price = swing_low * 0.99  # 1% below swing low
        
        size = int(round(risk_amount / (self.data.Close[-1] - stop_price)))
        if size > 0:
            fib_level = self._calculate_fib_levels(long=True)
            self.buy(size=size, 
                    sl=stop_price,
                    tp=fib_level,
                    tag="🌙 Lunar Long")
            print(f"🚀 MOON DEV LONG LAUNCH! Size: {size} | "
                  f"Entry: {self.data.Close[-1]:.2f} | "
                  f"SL: {stop_price:.2f} | TP: {fib_level:.2f} 🌕")

    def _enter_short(self):
        swing_high = self.swing_high[-1]
        risk_amount = self.equity * self.risk_pct
        stop_price = swing_high * 1.01  # 1% above swing high
        
        size = int(round(risk_amount / (stop_price - self.data.Close[-1])))
        if size > 0:
            fib_level = self._calculate_fib_levels(long=False)
            self.sell(size=size,
                     sl=stop_price,
                     tp=fib_level,
                     tag="🌑 Lunar Short")
            print(f"🌑 MOON DEV SHORT BLAST! Size: {size} | "
                  f"Entry: {self.data.Close[-1]:.2f} | "
                  f"SL: {stop_price:.2f} | TP: {fib_level:.2f} 🌘")
    
    def _calculate_fib_levels(self, long=True):
        swing_diff = self.swing_high[-1] - self.swing_low[-1]
        if long:
            return self.swing_low[-1] + swing_diff * 0.618
        return self.swing_high[-1] - swing_diff * 0.618

    def notify_trade(self, trade):
        if trade.is_closed:
            profit = trade.pl_pct * 100
            emoji = "🚀🌕" if profit > 0 else "💥🌑"
            print(f"✨ MOON DEV TRADE COMPLETE! {emoji} "
                  f"Profit: {profit:.2f}% ✨")

# Launch Moon Dev Back