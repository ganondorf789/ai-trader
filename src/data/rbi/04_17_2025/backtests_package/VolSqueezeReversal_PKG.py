I'll fix the backtesting.lib imports and crossunder/crossover usage while maintaining all the strategy logic. Here's the corrected code with Moon Dev themed debug prints:

```python
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np
from backtesting import Backtest, Strategy

# Load and preprocess data
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Verify funding rate column exists
if 'funding_rate' not in data.columns:
    raise ValueError("🌙🚨 CRITICAL: Funding rate data missing from dataset!")

class VolSqueezeReversal(Strategy):
    risk_pct = 1  # 1% risk per trade
    time_exit_bars = 12  # 12 candles = 3 hours (15m timeframe)
    
    def init(self):
        # Volatility Indicators
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, timeperiod=16, nbdevup=2, nbdevdn=2, matype=0,
            name=['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER']
        )
        self.bb_width = self.I(
            lambda: ((self.bb_upper - self.bb_lower)/self.bb_middle)*100,
            name='BB_WIDTH'
        )
        self.bb_width_ma = self.I(talib.SMA, self.bb_width, 5, name='BB_WIDTH_MA')
        
        # Historical Volatility
        self.hv_std = self.I(talib.STDDEV, self.data.Close, 20, name='HV_STD')
        self.hv = self.I(
            lambda: (self.hv_std/talib.SMA(self.data.Close, 20))*100,
            name='HV'
        )
        
        # Volume and Sentiment
        self.anchored_vwap = self.I(
            ta.vwap, high=self.data.High, low=self.data.Low, 
            close=self.data.Close, volume=self.data.Volume, anchor='D',
            name='ANCHORED_VWAP'
        )
        self.funding_rate = self.data['funding_rate']
        
        # Momentum Filter
        self.rsi = self.I(talib.RSI, self.data.Close, 14, name='RSI')
        self.swing_low = self.I(talib.MIN, self.data.Low, 5, name='SWING_LOW')
        
    def next(self):
        price = self.data.Close[-1]
        
        if self.position:
            # Volatility Expansion Exit
            if (self.bb_width[-1] > self.bb_width_ma[-1] and 
                self.data.Close[-2] > self.anchored_vwap[-2] and 
                self.data.Close[-1] < self.anchored_vwap[-1]):  # Replaced crossunder
                print(f"🌪️🌙 Volatility explosion! Closing at {price:.2f}")
                self.position.close()
            
            # Time-based Exit
            elif len(self.data) - self.position.entry_bar >= self.time_exit_bars:
                print(f"⌛🌙 Timed exit after {self.time_exit_bars} bars")
                self.position.close()
        else:
            # Long Entry Conditions
            if (self.bb_width[-1] < self.hv[-1] and
                self.funding_rate[-1] < 0 and
                price > self.anchored_vwap[-1] and
                self.rsi[-1] <= 60):
                
                # Risk Calculation
                stop_price = max(self.swing_low[-1], self.anchored_vwap[-1])
                risk_per_share = price - stop_price
                
                if risk_per_share > 0:
                    position_size = int(round((self.equity * self.risk_pct/100) / risk_per_share))
                    if position_size > 0:
                        print(f"🚀🌙 LAUNCH! Buying {position_size} @ {price:.2f}")
                        self.buy