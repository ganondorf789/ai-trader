Here's the fixed code with all backtesting.lib references removed and Moon Dev themed debug prints added:

```python
# 🌙 Moon Dev's VolatilitySurge Backtest Script ✨

import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

# Clean and prepare data 🌙
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'])
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}).set_index('datetime')

class VolatilitySurge(Strategy):
    bb_period = 20
    bb_dev = 2
    cmf_period = 21
    atr_period = 14
    sharpe_window = 10
    risk_pct = 0.01
    day_periods = 96  # 15min intervals in 24hrs
    
    def init(self):
        # 🌙✨ Core Indicators
        def bb_width(close):
            upper, middle, lower = talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)
            return (upper - lower)/middle
        
        self.bb_width = self.I(bb_width, self.data.Close, name='BB Width')
        self.bb_width_sma = self.I(talib.SMA, self.bb_width, self.bb_period, name='BB Width SMA')
        self.cmf = self.I(talib.CMF, self.data.High, self.data.Low, self.data.Close, self.data.Volume, 
                         self.cmf_period, name='CMF')
        self.prior_high = self.I(talib.MAX, self.data.High, self.day_periods, name='Prior High')
        self.prior_low = self.I(talib.MIN, self.data.Low, self.day_periods, name='Prior Low')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        # 🌙 Sharpe Ratio Calculation
        returns = self.I(talib.ROC, self.data.Close, 1, name='Returns')
        self.sma_ret = self.I(talib.SMA, returns, self.sharpe_window, name='SMA Returns')
        self.std_ret = self.I(talib.STDDEV, returns, self.sharpe_window, name='STD Returns')

    def next(self):
        # 🌙✨ Early Exit for Warmup
        if len(self.data) < max(self.bb_period, self.cmf_period, self.day_periods, self.atr_period) + 10:
            print("🌙 Warming up indicators... Not enough data yet!")
            return
        
        # 🌙 Current Values
        price = self.data.Close[-1]
        cmf = self.cmf[-1]
        bb_cond = self.bb_width[-1] < self.bb_width_sma[-1]
        atr = self.atr[-1]
        
        # 🌙✨ Sharpe Ratio Calculation
        sharpe = (self.sma_ret[-1]/self.std_ret[-1]) if self.std_ret[-1] != 0 else 0
        print(f"🌙 Current Sharpe Ratio: {sharpe:.2f} | BB Width: {self.bb_width[-1]:.4f}")

        # 🌙 Entry Logic
        if not self.position:
            # Long Entry 🚀
            if (bb_cond and cmf > 0.2 and 
                self.data.High[-1] > self.prior_high[-1]):
                sl = price - 1.5*atr
                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount/(price - sl))) if price > sl else 0
                if size:
                    self.buy(size=size, sl=sl)
                    print(f"🌙✨🚀 MOON DEV LONG SIGNAL! Size: {size} | Entry: {price:.2f} | SL: {sl:.2f} | ATR: {atr:.2f}")
            
            # Short Entry 🌑    
            elif (bb_cond and cmf < -0.2