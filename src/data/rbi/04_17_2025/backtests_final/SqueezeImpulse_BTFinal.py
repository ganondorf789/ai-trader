I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete corrected version with Moon Dev themed debug prints:

```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
from datetime import time

# Data preparation
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class SqueezeImpulse(Strategy):
    risk_pct = 0.01  # 1% per trade
    max_daily_loss = 0.05
    consecutive_loss_limit = 3
    
    def init(self):
        # Calculate indicators
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(talib.BBANDS, self.data.Close, 20, 2, 2)
        
        # Keltner Channels
        self.ema = self.I(talib.EMA, self.data.Close, 20, name='EMA')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 20, name='ATR')
        self.upper_kc = self.I(lambda: self.ema + 1.5*self.atr, name='Upper_KC')
        self.lower_kc = self.I(lambda: self.ema - 1.5*self.atr, name='Lower_KC')
        
        # Volume MA
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20, name='Volume_MA')
        
        # State tracking
        self.squeeze_history = []
        self.consecutive_losses = 0
        self.current_day = None
        self.daily_start_equity = None

    def next(self):
        # Moon Dev debug prints
        print(f"🌙 Processing {self.data.index[-1]} - Equity: ${self.equity:,.2f}")
        
        # Daily risk check
        current_day = self.data.index[-1].date()
        if current_day != self.current_day:
            self.current_day = current_day
            self.daily_start_equity = self.equity
            print(f"🌙✨ New day {current_day} - Starting equity: ${self.equity:,.2f}")
        
        if self.equity < self.daily_start_equity * (1 - self.max_daily_loss):
            print(f"🌙🚨 Daily loss limit triggered! No trades today")
            return
            
        # Consecutive loss check
        if self.consecutive_losses >= self.consecutive_loss_limit:
            print(f"🌙💥 3 consecutive losses! Trading paused")
            return
            
        # Time filter
        bar_time = self.data.index[-1].time()
        if time(0, 0) <= bar_time <= time(0, 29) or time(23, 30) <= bar_time <= time(23, 59):
            print(f"🌙⏰ Low liquidity period - Skipping trade")
            return
            
        # Squeeze detection
        current_squeeze = (self.bb_upper[-1] < self.upper_kc[-1]) and (self.bb_lower[-1] > self.lower_kc[-1])
        self.squeeze_history.append(current_squeeze)
        
        # Entry logic
        if not self.position:
            # Long entry conditions
            if self.data.Close[-1] > self.bb_upper[-1] and \
               self.data.Volume[-1] > 1.2 * self.volume_ma[-1] and \
               len(self.squeeze_history) >= 5 and \
               all(self.squeeze_history[-5:]):
                
                # Calculate position size
                sl = min(self.data.Low[-1], self.bb_lower[-1])
                risk_per_share = self.data.Close[-1] - sl
                if risk_per_share > 0:
                    size = int(round((self.equity * self.risk_pct) / risk_per_share))
                    if size > 0:
                        tp = self.data.Close[-1] + (self.upper_kc[-1] - self.bb