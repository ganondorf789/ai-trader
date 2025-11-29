Here's the complete fixed code with Moon Dev themed debug prints and all necessary corrections:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np
from pathlib import Path

# Data preparation - Moon Dev Cosmic Data Alignment 🌙✨
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
        # 🌌 Cosmic Bollinger Bands Calculation
        self.upper_band = self.I(lambda c: talib.BBANDS(c, 20, 2, 2, 0)[0], self.data.Close)
        self.middle_band = self.I(lambda c: talib.BBANDS(c, 20, 2, 2, 0)[1], self.data.Close)
        self.lower_band = self.I(lambda c: talib.BBANDS(c, 20, 2, 2, 0)[2], self.data.Close)
        
        # 🌊 Volume Surge Detection
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20)
        
        # 🌙 Track entry bar for lunar cycle exit timing
        self.entry_bar = 0
        print("🌙✨ Strategy Initialized - Cosmic Indicators Aligned! ✨🌙")

    def next(self):
        # Wait for sufficient cosmic data alignment
        if len(self.data) < 20:
            return
            
        # Calculate BBW and percentile - Moon Phase Analysis 🌓
        bbw = (self.upper_band - self.lower_band) / self.middle_band
        if len(bbw) < 20:
            return
            
        current_bbw = bbw[-1]
        bbw_window = bbw[-20:]
        bbw_percentile = (np.sum(bbw_window < current_bbw) / 20) * 100
        
        # Volume surge check - Cosmic Energy Spike Detection 🌠
        current_volume = self.data.Volume[-1]
        vol_surge = current_volume > 1.5 * self.volume_ma[-1]
        
        # Entry logic - Moon Dev Trading Protocol Activation
        if not self.position:
            # 🌕 Long entry - Riding the Lunar Wave
            if (bbw_percentile < 10 and
                vol_surge and
                self.data.Close[-1] > self.upper_band[-1]):
                
                # Cosmic Risk Management Protocol
                risk_pct = 0.01
                entry_price = self.data.Close[-1]
                stop_loss = self.middle_band[-1]
                risk_per_share = entry_price - stop_loss
                
                if risk_per_share > 0:
                    position_size = int(round((self.equity * risk_pct) / risk_per_share))
                    self.buy(size=position_size, sl=stop_loss)
                    self.entry_bar = len(self.data)
                    print(f"🌙✨🚀 COSMIC LONG SIGNAL! | Size: {position_size} | Entry: {entry_price:.2f} | SL: {stop_loss:.2f} | BBW%: {bbw_percentile:.1f}% | Moon Phase: Favorable 🌟")
            
            # 🌑 Short entry - Harnessing the Dark Moon Energy
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
                    print(f"🌙✨🔻 DARK MOON SHORT SIGNAL! | Size: {position_size} | Entry: {entry_price:.2f} | SL: {stop_loss:.