```python
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib

# Data preprocessing
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

class VolSurgeCross(Strategy):
    hv_5_period = 480  # 5 days in 15m intervals
    hv_20_period = 1920  # 20 days
    hv_3_period = 288  # 3 days
    vol_ma_period = 4800  # 50 days
    percentile_window = 200
    
    def init(self):
        # Calculate log returns
        close = self.data.Close.fillna(method='ffill')
        log_returns = np.log(close / close.shift(1))
        
        # Calculate historical volatility
        self.hv_5 = log_returns.rolling(self.hv_5_period).std()
        self.hv_20 = log_returns.rolling(self.hv_20_period).std()
        self.hv_3 = log_returns.rolling(self.hv_3_period).std()
        
        # Volume indicators
        self.vol_ma = self.I(talib.SMA, self.data.Volume, self.vol_ma_period)
        
        # 90th percentile calculation
        self.hv_3_90th = self.hv_3.rolling(self.percentile_window).quantile(0.9)
        
        # Add indicators
        self.I(lambda: self.hv_5, name='HV_5')
        self.I(lambda: self.hv_20, name='HV_20')
        self.I(lambda: self.hv_3, name='HV_3')
        self.I(lambda: self.hv_3_90th, name='HV3_90th')
    
    def next(self):
        if len(self.data) < max(self.hv_5_period, self.hv_20_period, self.vol_ma_period, self.percentile_window):
            return
            
        hv5, hv20, vol_ma = self.hv_5[-1], self.hv_20[-1], self.vol_ma[-1]
        current_vol, close = self.data.Volume[-1], self.data.Close[-1]
        
        if not self.position:
            # Long entry
            if crossover(hv20, hv5) and current_vol > 1.5*vol_ma:
                risk = 0.01 * self.equity
                sl_pct = 2 * hv5
                size = int(round(risk / (close * sl_pct)))
                if size:
                    self.buy(size=size, sl=close*(1-sl_pct))
                    print(f"🌙✨ MOON DEV LONG 🚀 | Size: {size} | Entry: {close:.2f} | SL%: {sl_pct:.4f}")
            
            # Short entry
            elif crossover(hv5, hv20) and current_vol > 1.5*vol_ma:
                risk = 0.01 * self.equity
                sl_pct = 2 * hv5
                size = int(round(risk / (close * sl_pct)))
                if size:
                    self.sell(size=size, sl=close*(1+sl_pct))
                    print(f"🌙✨ MOON DEV SHORT 💫 | Size: {size} | Entry: {close:.2f} | SL%: {sl_pct:.4f}")
        else:
            if self.hv_3[-1] > self.hv_3_90th[-1]:
                self.position.close()
                print(f"🌪️ VOL SURGE EXIT | Price: {close:.2f} | Current HV3: {self.hv_3[-1]:.4f} | 90th%: {self.hv_3_90th[-1]:.4f}")

bt = Backtest(data, VolSurgeCross, cash=1_000_000, exclusive_orders=True)
stats = bt.run()
print(stats