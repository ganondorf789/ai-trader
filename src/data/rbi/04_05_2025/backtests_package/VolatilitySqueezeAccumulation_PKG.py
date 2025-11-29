Here's the fixed code with all backtesting.lib references removed and Moon Dev themed debug prints added:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np

# Load and preprocess data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename columns to match required format
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Convert datetime and set index
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VolatilitySqueezeAccumulation(Strategy):
    bb_period = 20
    bb_std = 2
    kvo_fast = 34
    kvo_slow = 55
    kvo_signal = 13
    lookback = 200
    risk_pct = 0.01
    exit_days = 3
    bars_per_day = 96  # 15m intervals
    
    def init(self):
        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(talib.BBANDS, self.data.Close, 
                                                              timeperiod=self.bb_period, nbdevup=self.bb_std,
                                                              nbdevdn=self.bb_std, matype=0)
        
        # Bollinger Bandwidth
        self.bandwidth = self.I(lambda: (self.bb_upper - self.bb_lower) / self.bb_middle, name='Bandwidth')
        
        # Klinger Volume Oscillator
        kvo = ta.kvo(self.data.High, self.data.Low, self.data.Close, self.data.Volume, 
                    fast=self.kvo_fast, slow=self.kvo_slow, signal=self.kvo_signal)
        self.kvo_line = self.I(lambda: kvo.iloc[:,0].values, name='KVO')
        self.kvo_sig = self.I(lambda: kvo.iloc[:,1].values, name='KVO_Signal')
        
        # Swing High/Low
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20)
        
        # Historical bandwidth percentiles
        self.percentile_10 = self.I(lambda: pd.Series(self.bandwidth).rolling(self.lookback)
                                      .apply(lambda x: np.nanpercentile(x.dropna(), 10)), name='P10')
        self.percentile_50 = self.I(lambda: pd.Series(self.bandwidth).rolling(self.lookback)
                                      .apply(lambda x: np.nanpercentile(x.dropna(), 50)), name='P50')
        
        # ATR for trailing stop
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
        self.exit_bars = self.exit_days * self.bars_per_day

    def next(self):
        # Moon Dev Risk Management 🌙
        if not self.position:
            if len(self.data) < self.lookback or np.isnan(self.percentile_10[-1]):
                return
                
            bw_condition = (self.bandwidth[-1] < self.percentile_10[-1]) and (self.bandwidth[-1] < self.percentile_50[-1])
            kvo_condition = self.kvo_line[-1] > self.kvo_sig[-1]
            
            # Long Entry 🚀
            if bw_condition and kvo_condition and (self.data.Close[-2] < self.bb_upper[-2] and self.data.Close[-1] > self.bb_upper[-1]):
                sl = min(self.bb_lower[-1], self.swing_low[-1])
                risk_amount = self.equity * self.risk_pct
                risk_per_share = self.data.Close[-1] - sl
                
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    if size > 0:
                        self.buy(size=size, sl=sl, tag='Moon