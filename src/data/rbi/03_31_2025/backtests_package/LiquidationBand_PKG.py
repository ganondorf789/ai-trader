Here's the complete fixed code with all backtesting.lib references removed and Moon Dev themed debug prints added:

```python
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv', 
                   parse_dates=['datetime'], index_col='datetime')

# Clean data columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class LiquidationBand(Strategy):
    risk_pct = 0.01
    adr_period = 14
    volatility_threshold = 0.7
    
    def init(self):
        # Bollinger Bands using TA-Lib
        close = self.data.Close
        self.upper, self.middle, self.lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        self.I(lambda: self.upper, name='BB_upper')
        self.I(lambda: self.lower, name='BB_lower')
        
        # Volume SMA using TA-Lib
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
        # ADR tracking
        self.current_day = None
        self.daily_high = -np.inf
        self.daily_low = np.inf
        self.daily_ranges = []
        self.adr = 0

    def next(self):
        # Update daily range for ADR calculation
        current_dt = self.data.index[-1]
        if current_dt.date() != self.current_day:
            if self.current_day is not None:
                daily_range = self.daily_high - self.daily_low
                self.daily_ranges.append(daily_range)
                if len(self.daily_ranges) > self.adr_period:
                    self.daily_ranges.pop(0)
                if len(self.daily_ranges) >= self.adr_period:
                    self.adr = np.mean(self.daily_ranges[-self.adr_period:])
                    print(f"🌙 New ADR Calculated: {self.adr:.2f} ✨")
            
            self.current_day = current_dt.date()
            self.daily_high = self.data.High[-1]
            self.daily_low = self.data.Low[-1]
        else:
            self.daily_high = max(self.daily_high, self.data.High[-1])
            self.daily_low = min(self.daily_low, self.data.Low[-1])

        if not self.adr:
            return

        # Get current market conditions
        close = self.data.Close[-1]
        bb_lower = self.data.BB_lower[-1]
        bb_upper = self.data.BB_upper[-1]
        bandwidth = bb_upper - bb_lower
        
        # Entry conditions
        if not self.position:
            # Long entry (bullish rejection at lower band)
            long_cond = (
                self.data.Low[-1] <= bb_lower and
                self.data.Close[-1] > bb_lower and
                self.data.Close[-1] > self.data.Open[-1] and
                self.data.Volume[-1] > self.volume_sma[-1]
            )
            
            # Short entry (bearish rejection at upper band)
            short_cond = (
                self.data.High[-1] >= bb_upper and
                self.data.Close[-1] < bb_upper and
                self.data.Close[-1] < self.data.Open[-1] and
                self.data.Volume[-1] > self.volume_sma[-1]
            )
            
            if long_cond:
                sl = bb_lower - 1.5 * bandwidth
                risk = close - sl
                if risk > 0:
                    size = int(round((self.equity * self.risk_pct) / risk))
                    self.buy(size=size, sl=sl, tp=close + self.adr, tag={'entry_adr': self.adr})
                    print(f"🌙 BULLISH REJECTION DETECTED! LONG ENTRY @ {close:.2f} 🚀")
                    print(f"🌙 Stop Loss: {sl:.2f} | Take Profit