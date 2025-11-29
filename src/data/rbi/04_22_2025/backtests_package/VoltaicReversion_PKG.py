```python
from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import talib as ta
import pandas_ta as pta

# Data preparation
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
data = data.set_index('datetime')

class VoltaicReversion(Strategy):
    consecutive_losses = 0
    current_day = None
    daily_start_equity = None
    
    def init(self):
        # Calculate indicators using self.I()
        self.vwma = self.I(pta.vwma, self.data.Close, self.data.Volume, 20, name='VWMA')
        self.stddev = self.I(ta.STDDEV, self.data.Close, timeperiod=20, name='STDDEV')
        self.upper_2_5 = self.I(lambda v,s: v + 2.5*s, self.vwma, self.stddev, name='Upper2.5σ')
        self.lower_2_5 = self.I(lambda v,s: v - 2.5*s, self.vwma, self.stddev, name='Lower2.5σ')
        self.upper_3 = self.I(lambda v,s: v + 3*s, self.vwma, self.stddev, name='Upper3σ')
        self.lower_3 = self.I(lambda v,s: v - 3*s, self.vwma, self.stddev, name='Lower3σ')
        self.upper_1 = self.I(lambda v,s: v + s, self.vwma, self.stddev, name='Upper1σ')
        self.lower_1 = self.I(lambda v,s: v - s, self.vwma, self.stddev, name='Lower1σ')
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            ta.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name=['BB_U','BB_M','BB_L']
        )
        self.bb_width = self.I(lambda u,l,m: (u-l)/m, self.bb_upper, self.bb_lower, self.bb_middle, name='BB_WIDTH')
        self.vol_ma = self.I(ta.SMA, self.data.Volume, timeperiod=20, name='VolMA')

    def next(self):
        current_date = self.data.index[-1].date()
        if current_date != self.current_day:
            self.current_day = current_date
            self.daily_start_equity = self.equity
            print(f"🌞 MOON DEV: New day {current_date}, equity: {self.equity:.2f}")
            
        if self.daily_start_equity and self.equity < self.daily_start_equity * 0.95:
            print(f"🌧️ MOON DEV: Daily loss limit hit! No trades.")
            return
            
        if self.consecutive_losses >= 3:
            print(f"🌑 MOON DEV: 3 consecutive losses - trading paused")
            return
            
        if not self.position:
            price = self.data.Close[-1]
            vol = self.data.Volume[-1]
            bb_width = self.bb_width[-1]
            bb_width_prev = self.bb_width[-2] if len(self.bb_width) > 2 else 0
            
            if price > self.upper_2_5[-1] and bb_width < bb_width_prev and vol > self.vol_ma[-1]:
                stop = self.upper_3[-1]
                risk = abs(price - stop)
                size = int(round((self.equity * 0.01) / risk))
                if size > 0:
                    self.sell(size=size, sl=stop, tp=self.upper_1[-1])
                    print(f"🌙🚀 MOON DEV SHORT ENTRY! Size: {size} SL: {stop:.2f}")
                    
            elif price < self.lower_2_5[-1] and bb_width < bb_width_prev and vol > self.vol_ma[-1]:
                stop = self.lower_3[-1]