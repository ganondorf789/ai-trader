Here's the fixed code with all backtesting.lib references removed and proper indicator implementations:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np

# Data preparation
data = pd.read_csv("/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv")
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
data.set_index('datetime', inplace=True)

class VoltaicSqueeze(Strategy):
    risk_per_trade = 0.01
    max_daily_trades = 3
    
    def init(self):
        # Calculate indicators
        self.vwap = self.I(ta.vwap, high=self.data.High, low=self.data.Low, close=self.data.Close, volume=self.data.Volume)
        
        def bb_upper(close):
            upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return upper
        self.upper_band = self.I(bb_upper, self.data.Close)
        
        def bb_lower(close):
            _, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return lower
        self.lower_band = self.I(bb_lower, self.data.Close)
        
        def bb_middle(close):
            _, middle, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return middle
        self.middle_band = self.I(bb_middle, self.data.Close)
        
        self.bb_width = self.I(lambda u, l, m: (u - l)/m, self.upper_band, self.lower_band, self.middle_band)
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
        self.daily_trades = 0
        self.current_day = None

    def next(self):
        # Reset daily trade counter
        if self.data.index[-1].date() != self.current_day:
            self.current_day = self.data.index[-1].date()
            self.daily_trades = 0
        
        # Time filter (first 4 hours)
        if self.data.index[-1].hour >= 4:
            return
        
        # Don't trade without sufficient history
        if len(self.data) < 100:
            return
        
        # Entry logic
        if not self.position and self.daily_trades < self.max_daily_trades:
            # Volatility contraction check
            bb_range = self.bb_width[-50:-1]
            if len(bb_range) < 50:
                return
            contraction_level = np.percentile(bb_range, 20)
            
            # Consolidation check
            consolidated = all(
                abs(self.data.Close[-i] - self.vwap[-i])/self.vwap[-i] < 0.005 
                for i in range(2,7))
            
            # Volume surge check
            vol_ratio = self.data.Volume[-1]/self.volume_ma[-1]
            
            if (self.bb_width[-1] <= contraction_level and 
                consolidated and 
                vol_ratio >= 1.5):
                
                # Long entry condition
                if self.data.Close[-1] > self.upper_band[-1]:
                    risk_amount = self.equity * self.risk_per_trade
                    sl = self.lower_band[-1]
                    risk_per_share = self.data.Open[-1] - sl
                    size = int(round(risk_amount / risk_per_share))
                    
                    if size > 0:
                        self.buy(
                            size=size,
                            sl=sl,
                            tp=self.data.Open[-1] * (1 + 1.5*self.bb_width[-1])
                        self.daily_trades += 1
                        print(f"🌙🚀 MOON DEV LONG ENTRY | Size: {size} | Risk: {self.risk_per_trade*100:.0f}% | SL: {sl:.2f}")
                
                # Short entry condition
                elif self.data.Close[-1] < self.lower_band[-1]:
                    risk_amount = self.e