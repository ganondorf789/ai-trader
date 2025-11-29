```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
import numpy as np

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

class VolatilityReversal(Strategy):
    risk_pct = 0.01
    bb_period = 20
    bb_dev = 2
    bb_lookback = 200
    atr_period = 14
    supertrend_length = 10
    supertrend_multiplier = 3.0
    
    def init(self):
        # Calculate indicators
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, timeperiod=self.bb_period, 
                                          nbdevup=self.bb_dev, nbdevdn=self.bb_dev)
        self.I(lambda: upper, name='BB_Upper')
        self.I(lambda: lower, name='BB_Lower')
        bb_width = (upper - lower)/middle
        self.I(lambda: bb_width, name='BB_Width')
        self.bb_percentile = bb_width.rolling(self.bb_lookback).quantile(0.1)
        self.I(lambda: self.bb_percentile, name='BB_Perc')
        
        # Supertrend
        supertrend = ta.supertrend(high, low, close, 
                                 length=self.supertrend_length, 
                                 multiplier=self.supertrend_multiplier)
        st_dir = supertrend[f'SUPERTd_{self.supertrend_length}_{self.supertrend_multiplier}']
        self.I(lambda: st_dir, name='SuperTrend')
        
        # ATR
        atr = talib.ATR(high, low, close, self.atr_period)
        self.I(lambda: atr, name='ATR')
        
        self.trail_stop = None

    def next(self):
        # Moon Dev debugging
        if len(self.data) % 500 == 0:
            print(f"🌙 Processing bar {len(self.data)} - Equity: ${self.equity:,.0f} ✨")
            
        if len(self.data) < self.bb_lookback:
            return
            
        price = self.data.Close[-1]
        atr = self.data.ATR[-1]
        st = self.data.SuperTrend[-1]
        st_prev = self.data.SuperTrend[-2] if len(self.data.SuperTrend) > 1 else 0
        bb_width = self.data.BB_Width[-1]
        bb_perc = self.data.BB_Perc[-1]
        
        # Entry logic
        if not self.position:
            # Long entry
            if bb_width < bb_perc and st == 1 and st_prev == -1:
                risk_amount = self.risk_pct * self.equity
                size = int(round(risk_amount/(2*atr)))
                if size > 0:
                    self.buy(size=size, sl=price-2*atr)
                    self.trail_stop = price - 2*atr
                    print(f"🚀 MOON DEV LONG ENTRY 🌕 Size: {size} @ ${price:.2f} | ATR: {atr:.2f}")
            
            # Short entry
            elif bb_width < bb_perc and st == -1 and st_prev == 1:
                risk_amount = self.risk_pct * self.equity
                size = int(round(risk_amount/(2*atr)))
                if size > 0:
                    self.sell(size=size, sl=price+2*atr)
                    self.trail_stop = price + 2*atr
                    print(f"🌑 MOON DEV SHORT ENTRY 🌑 Size: {size} @ ${price:.2f} | ATR: {atr:.2f}")
        
        # Exit logic
        else: