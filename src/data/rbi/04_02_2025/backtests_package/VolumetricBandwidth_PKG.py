Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations, along with Moon Dev themed debug prints:

```python
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np
from backtesting import Strategy, Backtest

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and format data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolumetricBandwidth(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    
    def init(self):
        # Core indicators
        self.vwap = self.I(ta.vwap, 
                          high=self.data.High, 
                          low=self.data.Low,
                          close=self.data.Close,
                          volume=self.data.Volume,
                          anchor='D',
                          name='VWAP')
        
        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, 
            timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        
        # Volume and volatility indicators
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20)
        self.atr = self.I(talib.ATR, 
                         self.data.High, self.data.Low, self.data.Close, 20)
        
        # Bollinger Band Width calculation
        def bb_width(up, mid, low):
            return (up - low) / mid
        self.bb_width = self.I(bb_width, 
                              self.bb_upper, self.bb_middle, self.bb_lower,
                              name='BB_Width')
        
        self.entry_bar = 0  # For time-based exits

    def next(self):
        # Moon Dev Safety Checks 🌙
        if len(self.data) < 50:
            print("🌙 Moon Dev Alert: Not enough data points (need at least 50)")
            return
            
        # Session timing filter (first 2 hours)
        current_hour = self.data.index[-1].hour
        if not (0 <= current_hour < 2):
            print("🌙 Moon Dev Alert: Outside trading window (0-2 hours)")
            return
            
        # Volatility filter 🌊
        current_tr = self.data.High[-1] - self.data.Low[-1]
        if current_tr > 3 * self.atr[-1]:
            print("🌪️ Moon Dev Warning: Volatility too high, skipping trade")
            return
            
        # Historical percentile calculations
        lookback = 50
        hist_bb_width = self.bb_width[-lookback:-1]
        if len(hist_bb_width) < lookback-1:
            print("🌙 Moon Dev Alert: Not enough historical BB width data")
            return
            
        bb_90 = np.percentile(hist_bb_width, 90)
        bb_70 = np.percentile(hist_bb_width, 70)
        
        # Volume confirmation 🔊
        vol_ok = self.data.Volume[-1] > 1.5 * self.volume_ma[-1]
        
        # Entry Logic 🚀
        if not self.position:
            # VWAP crossover detection (replaced backtesting.lib.crossover)
            vwap_cross_above = (self.data.Close[-2] < self.vwap[-2] and 
                               self.data.Close[-1] > self.vwap[-1])
            vwap_cross_below = (self.vwap[-2] < self.data.Close[-2] and 
                                self.vwap[-1] > self.data.Close[-1])
            
            # Long entry conditions
            if all([vwap_cross_above,
                    vol_ok,
                    self.bb_width[-1] > bb_90]):
                
                risk_amount = self._broker.get_value() * self.risk_pct
                stop_price = self.bb_lower[-1]
                risk_per_share = self.data.Close[-1] - stop_price
                
                if risk_per_share > 0:
                    size = int(round(risk_amount /