Here's the complete backtesting implementation for the VolBand Breakout strategy:

```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import pandas_ta as ta

# Load and preprocess data
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolBandBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # 3-day VWAP (288 periods = 3 days * 96 periods/day)
        self.vwap = self.I(ta.vwap,
            high=self.data.High,
            low=self.data.Low,
            close=self.data.Close,
            volume=self.data.Volume,
            length=288,
            name='🌙 3D VWAP'
        )
        
        # Bollinger Bands (20, 2)
        self.upper_bb = self.I(
            lambda close: talib.BBANDS(close, 20, 2, 2)[0],
            self.data.Close,
            name='✨ Upper BB'
        )
        self.lower_bb = self.I(
            lambda close: talib.BBANDS(close, 20, 2, 2)[2],
            self.data.Close,
            name='✨ Lower BB'
        )
        
        # Volume indicators (20-day = 1920 periods)
        self.volume_avg = self.I(talib.SMA, self.data.Volume, 1920, name='📊 Vol Avg')
        self.volume_std = self.I(talib.STDDEV, self.data.Volume, 1920, name='📈 Vol Std')
        
    def next(self):
        if len(self.data) < 20:
            return
            
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        volume_threshold = self.volume_avg[-1] + 2*self.volume_std[-1]
        
        # Calculate VWAP convergence
        vwap_converging = False
        if len(self.vwap) >= 3:
            vwap_change = (self.vwap[-1] - self.vwap[-3]) / self.vwap[-3] * 100
            vwap_converging = abs(vwap_change) <= 0.5  # 0.5% max change over 3 periods
        
        # Long entry conditions
        if (not self.position and
            current_close > self.upper_bb[-1] and
            vwap_converging and
            current_volume > volume_threshold):
            
            risk_amount = self.equity * self.risk_per_trade
            stop_price = self.lower_bb[-1] * 0.995  # 0.5% buffer
            risk_per_share = current_close - stop_price
            
            if risk_per_share > 0:
                size = int(round(risk_amount / risk_per_share))
                self.buy(size=size, sl=stop_price)
                print(f"🚀🌕 MOON DEV LONG: {size} units @ {current_close:.2f} | SL: {stop_price:.2f}")
        
        # Short entry conditions
        elif (not self.position and
            current_close < self.lower_bb[-1] and
            vwap_converging and
            current_volume > volume_threshold):
            
            risk_amount = self.equity * self.risk_per_trade
            stop_price = self.upper_bb[-1] * 1.005  # 0.5% buffer
            risk_per_share = stop_price - current_close
            
            if risk_per_share > 0:
                size = int(round(risk_amount / risk_per_share))
                self.sell(size=size, sl=stop_price)
                print(f"🌑🚀 MOON DEV SHORT: {size} units @ {current_close:.2f} | SL: {stop_price:.2f}")
        
        # Exit conditions
        if self.position.is_long and current_close < self.upper_bb[-1]:
            self.position.close()
            print(f"🌕✨ MOON DEV EXIT LONG: {current