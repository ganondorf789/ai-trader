```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib
import numpy as np
from pathlib import Path

# Data preparation
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class VolSurgeBreakout(Strategy):
    def init(self):
        # Bollinger Bands
        self.upper_band = self.I(lambda c: talib.BBANDS(c, 20, 2, 2, 0)[0], self.data.Close)
        self.middle_band = self.I(lambda c: talib.BBANDS(c, 20, 2, 2, 0)[1], self.data.Close)
        self.lower_band = self.I(lambda c: talib.BBANDS(c, 20, 2, 2, 0)[2], self.data.Close)
        
        # Volume MA
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20)
        
        # Track entry bar for time exit
        self.entry_bar = 0

    def next(self):
        # Wait for sufficient data
        if len(self.data) < 20:
            return
            
        # Calculate BBW and percentile
        bbw = (self.upper_band - self.lower_band) / self.middle_band
        if len(bbw) < 20:
            return
            
        current_bbw = bbw[-1]
        bbw_window = bbw[-20:]
        bbw_percentile = (np.sum(bbw_window < current_bbw) / 20) * 100
        
        # Volume surge check
        current_volume = self.data.Volume[-1]
        vol_surge = current_volume > 1.5 * self.volume_ma[-1]
        
        # Entry logic
        if not self.position:
            # Long entry
            if (bbw_percentile < 10 and
                vol_surge and
                self.data.Close[-1] > self.upper_band[-1]):
                
                # Risk management
                risk_pct = 0.01
                entry_price = self.data.Close[-1]
                stop_loss = self.middle_band[-1]
                risk_per_share = entry_price - stop_loss
                
                if risk_per_share > 0:
                    position_size = int(round((self.equity * risk_pct) / risk_per_share))
                    self.buy(size=position_size, sl=stop_loss)
                    self.entry_bar = len(self.data)
                    print(f"🌙✨🚀 LONG Entry! | Size: {position_size} | Entry: {entry_price:.2f} | SL: {stop_loss:.2f} | BBW%: {bbw_percentile:.1f}% 🌟")
            
            # Short entry
            elif (bbw_percentile < 10 and
                  vol_surge and
                  self.data.Close[-1] < self.lower_band[-1]):
                  
                risk_pct = 0.01
                entry_price = self.data.Close[-1]
                stop_loss = self.middle_band[-1]
                risk_per_share = stop_loss - entry_price
                
                if risk_per_share > 0:
                    position_size = int(round((self.equity * risk_pct) / risk_per_share))
                    self.sell(size=position_size, sl=stop_loss)
                    self.entry_bar = len(self.data)
                    print(f"🌙✨🔻 SHORT Entry! | Size: {position_size} | Entry: {entry_price:.2f} | SL: {stop_loss:.2f} | BBW%: {bbw_percentile:.1f}% 🌟")

        # Exit logic
        if self.position.is_long:
            # Opposite band exit
            if self.data.Low[-1] <= self.lower_band[-1]:
                self.position.close()
                print(f"🌙📉 LONG Exit! | Price: {self.data.Close[-1]:.2f} | Profit: {self.position.pl_pct:.2f}% 🌌")