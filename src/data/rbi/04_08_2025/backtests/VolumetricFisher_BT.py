```python
import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Load and preprocess data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VolumetricFisher(Strategy):
    risk_percent = 0.01
    swing_period = 20
    
    def init(self):
        # Calculate Fisher Transform components
        median_price = (self.data.High + self.data.Low) / 2
        hh_median = self.I(talib.MAX, median_price, timeperiod=9, name='HH_MED')
        ll_median = self.I(talib.MIN, median_price, timeperiod=9, name='LL_MED')
        
        normalized = (median_price - ll_median) / (hh_median - ll_median)
        normalized = np.clip(normalized, 0.01, 0.99)
        self.fisher = self.I(lambda: 0.5 * np.log((1 + normalized)/(1 - normalized)), name='FISHER')
        self.fisher_signal = self.I(talib.EMA, self.fisher, timeperiod=9, name='FISHER_SIG')
        
        # Calculate Volume-Weighted MACD
        vw_price = self.data.Close * self.data.Volume
        self.macd = self.I(talib.MACD, vw_price, fastperiod=12, slowperiod=26, signalperiod=9)[0]
        self.macd_signal = self.I(talib.MACD, vw_price, fastperiod=12, slowperiod=26, signalperiod=9)[1]
        
        # Parabolic SAR
        self.sar = self.I(talib.SAR, self.data.High, self.data.Low, acceleration=0.02, maximum=0.2, name='SAR')
        
        # Swing levels for stop loss
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period, name='SWING_LOW')
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period, name='SWING_HIGH')
        
    def next(self):
        if len(self.data) < 50:  # Warmup period
            return
            
        current_close = self.data.Close[-1]
        equity = self.equity
        
        # Long entry logic
        if (crossover(self.fisher, self.fisher_signal) and
            self.data.Close[-2] > self.data.Close[-1] and
            self.macd[-2] < self.macd[-1] and
            not self.position):
            
            sl_price = self.swing_low[-1]
            risk_amount = equity * self.risk_percent
            risk_per_share = current_close - sl_price
            if risk_per_share > 0:
                size = int(round(risk_amount / risk_per_share))
                print(f"🌙✨ BULLISH CROSS detected! Buying {size} shares at {current_close} 🚀")
                self.buy(size=size, sl=sl_price, tag='long_entry')
        
        # Short entry logic
        elif (crossunder(self.fisher, self.fisher_signal) and
              self.data.Close[-2] < self.data.Close[-1] and
              self.macd[-2] > self.macd[-1] and
              not self.position):
            
            sl_price = self.swing_high[-1]
            risk_amount = equity * self.risk_percent
            risk_per_share = sl_price - current_close
            if risk_per_share > 0:
                size = int(round(risk_amount / risk_per_share))
                print(f"🌙✨ BEARISH CROSS detected! Shorting {size} shares at {current_close} 🌑")
                self.sell(size=size, sl=sl_price, tag='short_entry')
        
        # Exit logic
        for trade in self.trades:
            if trade.is_long and self.sar[-1]