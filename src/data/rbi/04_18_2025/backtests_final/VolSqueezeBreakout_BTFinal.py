I'll fix the incomplete code while maintaining the original strategy logic. Here's the debugged version with Moon Dev themed improvements:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Load and preprocess data with Moon Dev precision 🌙
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

print("🌙 DATA LOADED SUCCESSFULLY! Beginning lunar analysis...")

class VolSqueezeBreakout(Strategy):
    risk_percent = 0.02  # Fixed to proper fraction for percentage sizing
    tp_multiplier = 2
    
    def init(self):
        # Bollinger Bands with lunar precision 🌕
        close = self.data.Close.values.astype(float)
        self.upper, self.middle, self.lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        self.bb_upper = self.I(lambda: self.upper, name='Upper BB')
        self.bb_lower = self.I(lambda: self.lower, name='Lower BB')
        
        # BB Width calculations
        self.bb_width = (self.upper - self.lower) / self.middle
        self.bb_width_ind = self.I(lambda: self.bb_width, name='BB Width')
        
        # Historical percentile
        bb_width_series = pd.Series(self.bb_width)
        self.bb_percentile = bb_width_series.expanding().quantile(0.1).values
        self.bb_percentile_ind = self.I(lambda: self.bb_percentile, name='BB Percentile')
        
        # Volume indicators
        volume = self.data.Volume.values.astype(float)
        self.volume_sma = self.I(talib.SMA, volume, timeperiod=20, name='Volume SMA')
        
        # Risk management
        self.consecutive_losses = 0
        self.last_date = None
        self.daily_start_equity = None
        self.daily_loss_limit_hit = False
    
    def next(self):
        # Daily loss check
        current_date = self.data.index[-1].date()
        if self.last_date != current_date:
            self.last_date = current_date
            self.daily_start_equity = self.equity
            self.daily_loss_limit_hit = False
            print(f"🌙 NEW MOON CYCLE STARTED: {current_date} | Starting Equity: {self.daily_start_equity:.2f}")
            
        if self.daily_loss_limit_hit:
            return
        
        if self.daily_start_equity and (self.daily_start_equity - self.equity)/self.daily_start_equity >= 0.05:
            print("🌙🚨 DAILY LOSS LIMIT TRIGGERED! Closing all positions.")
            self.position.close()
            self.daily_loss_limit_hit = True
            return
        
        # Check consecutive losses
        if self.consecutive_losses >= 3:
            print("🌙⚠️ CONSECUTIVE LOSS LIMIT REACHED! Skipping trades today.")
            return
        
        # Entry logic
        if not self.position:
            current_bb_width = self.bb_width_ind[-1]
            current_percentile = self.bb_percentile_ind[-1]
            volume_spike = self.data.Volume[-1] > self.volume_sma[-1] * 1.15
            
            if current_bb_width < current_percentile and volume_spike:
                if self.data.Close[-1] > self.bb_upper[-1]:
                    print(f"🌙🚀 BULLISH BREAKOUT DETECTED! Entering long at {self.data.Close[-1]}")
                    # Fixed position sizing to use proper fraction
                    self.buy(size=self.risk_percent, 
                            sl=self.bb_lower[-1], 
                            tp=self.data.Close[-1] + (self.bb_width[-1] * self.tp_multiplier))
                elif self.data.Close[-1] < self.bb_lower[-1]:
                    print(f"🌙🌑