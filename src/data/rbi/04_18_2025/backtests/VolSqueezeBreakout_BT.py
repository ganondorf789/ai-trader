```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Load and preprocess data
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

class VolSqueezeBreakout(Strategy):
    risk_percent = 0.01
    tp_multiplier = 2
    
    def init(self):
        # Bollinger Bands
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
            
        if self.daily_loss_limit_hit:
            return
        
        if self.daily_start_equity and (self.daily_start_equity - self.equity)/self.daily_start_equity >= 0.05:
            print("🌙🚨 DAILY LOSS LIMIT TRIGGERED! Closing all positions.")
            self.position.close()
            self.daily_loss_limit_hit = True
            return
        
        # Check consecutive losses
        if self.consecutive_losses >= 3:
            return
        
        # Entry logic
        if not self.position:
            current_bb_width = self.bb_width_ind[-1]
            current_percentile = self.bb_percentile_ind[-1]
            volume_spike = self.data.Volume[-1] > self.volume_sma[-1] * 1.15
            
            if current_bb_width < current_percentile and volume_spike:
                if self.data.Close[-1] > self.bb_upper[-1]:
                    self.enter_long()
                elif self.data.Close[-1] < self.bb_lower[-1]:
                    self.enter_short()
        
        # Exit logic
        else:
            if self.position.is_long:
                if self.data.Close[-1] < self.bb_upper[-1]:
                    print(f"🌙✨ RE-ENTERED BANDS! Closing long at {self.data.Close[-1]}")
                    self.position.close()
            elif self.position.is_short:
                if self.data.Close[-1] > self.bb_lower[-1]:
                    print(f"🌙✨ RE-ENTERED BANDS! Closing short at {self.data.Close[-1]}")
                    self.position.close()
    
    def enter_long(self):
        entry_price = self.data.Close[-1]
        sl = self.bb_lower[-1]
        risk = entry_price - sl
        if risk <= 0: return
        
        size = int(round((self.equity * self.risk_percent) / risk))
        if size > 0:
            tp = entry_price +