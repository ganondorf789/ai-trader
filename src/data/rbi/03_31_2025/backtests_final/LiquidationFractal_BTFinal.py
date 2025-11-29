Here's the debugged version of your Liquidation Fractal backtest code with Moon Dev themed improvements:

```python
# 🌙 Moon Dev's Liquidation Fractal Backtest 🌙
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# ========= DATA PREPARATION =========
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and prepare data 🌐
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class LiquidationFractal(Strategy):
    # 🌙 Strategy Parameters
    atr_period = 14
    swing_window = 20
    risk_pct = 0.01
    max_trades = 3
    fib_cluster_threshold = 0.005  # 0.5%
    
    def init(self):
        # 🌌 Core Indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, 
                         self.data.Close, self.atr_period)
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_window)
        
        # 🌠 Additional Metrics
        self._prev_equity = self._equity = [self.equity]

    def next(self):
        # 🌕 Moon Dev Debug Header
        print(f"\n🌙✨ Processing {self.data.index[-1]} ✨🌙")
        
        # Skip if not enough data
        if len(self.data.Close) < self.swing_window * 2:
            return
            
        current_close = self.data.Close[-1]
        atr = self.atr[-1]
        sh = self.swing_high[-1]
        sl = self.swing_low[-1]
        
        # ========= FIBONACCI CLUSTER CALCULATION =========
        fib_levels = []
        if sh > sl:  # Valid swing range
            price_range = sh - sl
            fib_levels = [
                sh - 0.382 * price_range,
                sh - 0.5 * price_range,
                sh - 0.618 * price_range
            ]
        
        # ========= LIQUIDATION ZONE DETECTION =========
        long_zone = False
        short_zone = False
        
        # Long Zone Detection (price near swing low)
        if current_close <= sl * (1 + self.fib_cluster_threshold):
            fib_hits = sum(1 for f in fib_levels if abs(f - sl)/sl <= self.fib_cluster_threshold)
            if fib_hits >= 2:
                long_zone = True
                print(f"🚀 BULLISH ZONE DETECTED! Swing Low: {sl:.2f} | Fib Cluster: {fib_levels}")
        
        # Short Zone Detection (price near swing high)
        if current_close >= sh * (1 - self.fib_cluster_threshold):
            fib_hits = sum(1 for f in fib_levels if abs(f - sh)/sh <= self.fib_cluster_threshold)
            if fib_hits >= 2:
                short_zone = True
                print(f"🌑 BEARISH ZONE DETECTED! Swing High: {sh:.2f} | Fib Cluster: {fib_levels}")
        
        # ========= ENTRY LOGIC =========
        if long_zone and self._is_bullish_rejection():
            print(f"🌟 BULLISH REJECTION CONFIRMED!")
            self._enter_long(atr)
            
        if short_zone and self._is_bearish_rejection():
            print(f"🌟 BEARISH REJECTION CONFIRMED!")
            self._enter_short(atr)
            
        # ========= RISK MANAGEMENT =========
        self._manage_positions(atr)
        self._check_daily_loss_limit()

    def _is_bullish_rejection(self):
        # Detect pinbar or bullish engulfing
        body = abs(self.data.Close[-1] - self.data.Open[-1])
        lower_wick = min(self.data.Open[-1], self.data.Close[-1]) - self.data