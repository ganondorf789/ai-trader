Here's the fixed code with all backtesting.lib imports removed and proper indicator implementations:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np

# Load and prepare data
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class ForceBandContraction(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Calculate indicators using proper self.I() wrapper
        self.efi = self.I(ta.force_index, self.data.Close, self.data.Volume, length=13)
        self.efi_sma3 = self.I(talib.SMA, self.efi, timeperiod=288)  # 3-day SMA for 15m data
        
        # Bollinger Bands
        self.bb_upper = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0], self.data.Close)
        self.bb_middle = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1], self.data.Close)
        self.bb_lower = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2], self.data.Close)
        
        # Bandwidth calculations
        self.band_width = self.I(lambda u, l, m: (u - l)/m, self.bb_upper, self.bb_lower, self.bb_middle)
        self.band_width_avg = self.I(talib.SMA, self.band_width, timeperiod=20)
        
        # Volatility measures
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        # Trade tracking
        self.trades_today = 0
        self.current_date = None
        self.entry_bar = 0
        self.stop_loss = 0

    def next(self):
        # Reset daily trade counter
        current_date = self.data.index[-1].date()
        if current_date != self.current_date:
            self.current_date = current_date
            self.trades_today = 0
            print(f"🌙 New Moon Day: {current_date.strftime('%Y-%m-%d')} ✨")

        # Skip if max daily trades reached
        if self.trades_today >= 3:
            return

        # Calculate current conditions - replaced crossover with manual check
        efi_cross = (self.efi[-2] < self.efi_sma3[-2] and self.efi[-1] > self.efi_sma3[-1])
        bw_contraction = self.band_width[-1] < self.band_width_avg[-1]
        atr_valid = (self.atr[-1] / self.data.Close[-1]) >= 0.005

        # Entry logic
        if not self.position and atr_valid:
            if efi_cross and bw_contraction:
                # Long entry condition
                if self.data.Close[-1] > self.bb_upper[-1]:
                    self.calculate_position_size(long=True)
                    self.trades_today += 1
                    print(f"🚀🌙 Moon Boost: LONG Signal Confirmed! Band Width {self.band_width[-1]:.4f}")
                
                # Short entry condition
                elif self.data.Close[-1] < self.bb_lower[-1]:
                    self.calculate_position_size(long=False)
                    self.trades_today += 1
                    print(f"🌑🌙 Moon Shadow: SHORT Signal Activated! BW {self.band_width[-1]:.4f}")

        # Exit logic
        if self.position:
            bars_held = len(self.data) - self.entry_bar
            price = self.data.Close[-1]

            if self.position.is_long: