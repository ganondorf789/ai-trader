```python
# 🌙 Moon Dev's VolumetricVwap Backtest 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import datetime

# Data preparation 🌐
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and prepare data columns �
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

class VolumetricVwap(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌡️
    
    def init(self):
        # Calculate VWAP using cumulative method 🌗
        typical_price = (self.data.High + self.data.Low + self.data.Close) / 3
        cumulative_tpv = (typical_price * self.data.Volume).cumsum()
        cumulative_vol = self.data.Volume.cumsum()
        self.vwap = cumulative_tpv / cumulative_vol
        self.I(lambda: self.vwap, name='VWAP')
        
        # Volume 90th percentile (20-period) 📊
        self.volume_90th = self.data.Volume.rolling(20).quantile(0.9)
        self.I(lambda: self.volume_90th, name='Volume_90th')
        
        # Bollinger Bands with %B calculation 📉
        self.I(lambda c: talib.BBANDS(c, 20, 2, 2)[0], self.data.Close, name='BB_upper')
        self.I(lambda c: talib.BBANDS(c, 20, 2, 2)[2], self.data.Close, name='BB_lower')
        
    def next(self):
        # Wait for sufficient data 🌌
        if len(self.data) < 20:
            return
            
        # Time-based entry cutoff ⏰
        current_time = self.data.index[-1].time()
        if current_time >= datetime.time(14, 30) and not self.position:
            return
            
        # Entry logic 🚪
        if not self.position:
            prev_idx = -2  # Previous bar index
            
            # Long entry conditions 🌕
            if (crossover(self.data.Close[prev_idx], self.vwap[prev_idx]) and \
               self.data.Volume[prev_idx] > self.volume_90th[prev_idx]:
                
                # Risk management calculations 🛡️
                entry_price = self.data.Open[-1]
                lookback = 5
                start_idx = max(0, len(self.data) - lookback - 1)
                stop_loss = min(self.data.Low[start_idx:-1])
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = entry_price - stop_loss
                
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    self.buy(size=size, sl=stop_loss)
                    print(f"🌙✨ LONG ENTRY: {size} units @ {entry_price} | SL {stop_loss} 🚀")
            
            # Short entry conditions 🌑
            elif (crossover(self.vwap[prev_idx], self.data.Close[prev_idx]) and \
                 self.data.Volume[prev_idx] > self.volume_90th[prev_idx]:
                
                entry_price = self.data.Open[-1]
                lookback = 5
                start_idx = max(0, len(self.data) - lookback - 1)
                stop_loss = max(self.data.High[start_idx:-1])
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = stop_loss - entry_price
                
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    self.sell(size=size, sl=stop_loss)
                    print(f"🌙✨ SHORT ENTRY: {size} units @ {entry_price} | SL {stop_loss} 🚀")
        
        # Exit logic using %B 🌗
        if self.position:
            current_close = self.data.Close[-1]
            upper = self.data.BB_upper[-1]
            lower = self.data.BB_lower[-