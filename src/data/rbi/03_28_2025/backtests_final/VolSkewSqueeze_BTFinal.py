I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete, debugged version with Moon Dev themed prints:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Load and prepare data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Clean and format columns
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VolSkewSqueeze(Strategy):
    bb_period = 20
    bb_dev = 2
    atr_period = 14
    volume_ma_period = 20
    squeeze_lookback = 100
    risk_pct = 0.01
    trailing_atr_mult = 2
    max_daily_trades = 3
    exit_bars = 5

    def init(self):
        # Bollinger Bands
        self.bb_upper, _, self.bb_lower = self.I(talib.BBANDS, self.data.Close, 
                                                timeperiod=self.bb_period, nbdevup=self.bb_dev,
                                                nbdevdn=self.bb_dev, matype=0)
        
        # Band width calculation
        self.band_width = self.I(lambda u, l: u - l, self.bb_upper, self.bb_lower)
        
        # Squeeze detection
        self.squeeze_threshold = self.I(talib.MIN, self.band_width, self.squeeze_lookback)
        
        # Volume indicators
        self.vol_ma = self.I(talib.SMA, self.data.Volume, self.volume_ma_period)
        
        # Volatility indicator
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        self.trade_count = 0
        self.current_day = None
        self.entry_bar = 0

        print("🌙✨ Moon Dev Indicators Initialized! Ready for cosmic trading...")

    def next(self):
        # Reset daily trade counter
        if self.data.index[-1].date() != self.current_day:
            self.current_day = self.data.index[-1].date()
            self.trade_count = 0
            print(f"🌙📅 New Trading Day: {self.current_day} | Trade counter reset")

        if self.trade_count >= self.max_daily_trades:
            print("🌙🚫 Daily trade limit reached! No more trades today")
            return

        price = self.data.Close[-1]
        atr_val = self.atr[-1]
        vol_ratio = self.data.Volume[-1]/self.vol_ma[-1]

        # Exit conditions
        if self.position:
            self.entry_bar += 1
            
            # Time-based exit
            if self.entry_bar >= self.exit_bars:
                self.position.close()
                print("🌙⏰ Time-based exit triggered")
                return
                
            # Trailing stop
            if self.position.is_long:
                trailing_stop = price - (self.trailing_atr_mult * atr_val)
                self.position.sl = max(trailing_stop, self.position.sl or 0)
            else:
                trailing_stop = price + (self.trailing_atr_mult * atr_val)
                self.position.sl = min(trailing_stop, self.position.sl or float('inf'))
            return

        # Long entry
        if (price > self.bb_upper[-1] and
            vol_ratio >= 1.5 and
            self.band_width[-1] > self.squeeze_threshold[-1]):
            
            sl = price - atr_val
            risk_amount = self.equity * self.risk_pct
            position_size = int(round(risk_amount/(price - sl)))
            
            self.buy(size=position_size, sl=sl,
                    tag="🌙🚀 LONG! VolSkew Confirmed | BB Breakout")
            self.trade_count += 1
            self.entry_bar = 0
            print(f"🌙💫 LONG ENTRY at {price:.2f} | SL: {sl