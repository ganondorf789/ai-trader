```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Load and clean data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolSqueezeReversion(Strategy):
    risk_per_trade = 0.01
    max_positions = 3
    
    def init(self):
        # Bollinger Bands
        self.upper_bb = self.I(lambda c: talib.BBANDS(c, 20, 2, 2)[0], self.data.Close)
        self.lower_bb = self.I(lambda c: talib.BBANDS(c, 20, 2, 2)[2], self.data.Close)
        self.bb_width = self.I(lambda u, l: (u - l) / ((u + l)/2), self.upper_bb, self.lower_bb)
        self.bb_width_avg = self.I(talib.SMA, self.bb_width, 20)
        
        # VWAP and channels
        def calc_vwap(h, l, c, v):
            return ta.vwap(high=h, low=l, close=c, volume=v, anchor='D')
        self.vwap = self.I(calc_vwap, self.data.High, self.data.Low, self.data.Close, self.data.Volume)
        self.vwap_std = self.I(talib.STDDEV, self.data.Close, 20)
        self.upper_2sigma = self.I(lambda v, s: v + 2*s, self.vwap, self.vwap_std)
        self.lower_2sigma = self.I(lambda v, s: v - 2*s, self.vwap, self.vwap_std)
        self.upper_3sigma = self.I(lambda v, s: v + 3*s, self.vwap, self.vwap_std)
        self.lower_3sigma = self.I(lambda v, s: v - 3*s, self.vwap, self.vwap_std)
        
        # Volume analysis
        self.vol_avg = self.I(talib.SMA, self.data.Volume, 10)
    
    def next(self):
        # Skip first/last 30 minutes
        hour = self.data.index[-1].hour
        minute = self.data.index[-1].minute
        if (hour == 0 and minute < 30) or (hour == 23 and minute >= 30):
            return
        
        # Calculate position size
        equity = self.equity
        risk_amount = equity * self.risk_per_trade
        
        # Long entry logic
        if (not self.position and
            self.bb_width[-1] < self.bb_width_avg[-1] and
            self.data.Close[-1] > self.upper_2sigma[-1] and
            self.data.Volume[-1] > 1.2 * self.vol_avg[-1] and
            len(self.trades) < self.max_positions):
            
            sl_price = self.lower_3sigma[-1]
            risk_per_unit = self.data.Close[-1] - sl_price
            position_size = int(round(risk_amount / risk_per_unit))
            
            if position_size > 0:
                self.buy(size=position_size, sl=sl_price, tag={'entry_width': self.bb_width[-1]})
                print(f"🚀 MOON DEV LONG LAUNCH: {self.data.Close[-1]:.2f} | Size: {position_size} | SL: {sl_price:.2f} 🌙")
        
        # Short entry logic
        elif (not self.position and
              self.bb_width[-1] < self.bb_width_avg[-1] and
              self.data.Close[-1] < self.lower_2sigma[-1] and
              self.data.Volume[-1] > 1.2 * self.vol_avg[-1] and
              len(self.trades) < self.max_positions):
            
            sl_price = self.upper_3sigma[-1]
            risk_per_unit = sl_price - self.data