Here's the complete fixed code with Moon Dev themed debug prints and proper position sizing:

```python
# Required imports - CLEANED from backtesting.lib 🌙
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

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

class LiquiSqueezeReversal(Strategy):
    risk_pct = 0.01
    bb_period = 80  # 20-periods on 1h (15m*4)
    squeeze_ma_period = 20
    zscore_lookback = 100
    surge_threshold = 1.28  # Top 10%
    
    def init(self):
        # Bollinger Bands with 1h equivalent period
        close = self.data.Close
        self.upper, self.middle, self.lower = self.I(talib.BBANDS, close, 
                                                   timeperiod=self.bb_period,
                                                   nbdevup=2, nbdevdn=2,
                                                   matype=0)
        
        # Volatility squeeze detection
        self.bandwidth = self.upper - self.lower
        self.bandwidth_ma = self.I(talib.SMA, self.bandwidth, 
                                  timeperiod=self.squeeze_ma_period)
        
        # Liquidation surge proxy (volume z-score)
        self.volume_ma = self.I(talib.SMA, self.data.Volume,
                               timeperiod=self.zscore_lookback)
        self.volume_std = self.I(talib.STDDEV, self.data.Volume,
                                timeperiod=self.zscore_lookback)
        self.z_score = (self.data.Volume - self.volume_ma) / self.volume_std
        
        # Pattern recognition
        self.engulfing = self.I(talib.CDLENGULFING, self.data.Open, self.data.High,
                               self.data.Low, self.data.Close)

    def next(self):
        if self.position:
            return  # Exit if position exists

        # Current price action
        o, h, l, c = self.data.Open[-1], self.data.High[-1], self.data.Low[-1], self.data.Close[-1]
        body_size = abs(c - o)
        total_range = h - l
        
        # 1. Bollinger Squeeze Condition
        squeeze_active = (self.bandwidth[-1] < 0.9 * self.bandwidth_ma[-1])
        
        # 2. Liquidation Surge Alert 🚨
        liquidity_surge = self.z_score[-1] > self.surge_threshold
        
        # 3. Rejection Candle Detection
        bullish_reject = (self.engulfing[-1] == 100) or (
            (min(c, o) - l > 2 * body_size) and 
            ((min(c, o) - l) > 0.6 * total_range)
        
        bearish_reject = (self.engulfing[-1] == -100) or (
            (h - max(c, o) > 2 * body_size) and 
            ((h - max(c, o)) > 0.6 * total_range))

        # Moon Dev Signal Validation 🌙✨
        if squeeze_active and liquidity_surge:
            if bullish_reject:
                sl = l - 0.1 * (h - l)  # Stop below candle low
                risk_amount = self.equity * self.risk_pct
                risk_per_unit = c - sl
                units = int(round(risk_amount / risk_per_unit))
                
                print(f"🌕 MOON DEV LONG SIGNAL ACTIVATED 🌙\n"
                      f"✨ Entry: {c:.2f} | 🛡️ SL: {sl:.2f}\n"
                      f"📊 Position Size: {units} units\n"
                      f"💎 Risk: {self.risk_pct*100:.1f}% of equity")
                
                self.buy(size=units, sl=sl)
                
            elif bearish_reject:
                sl