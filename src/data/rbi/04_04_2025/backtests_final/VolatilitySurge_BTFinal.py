I'll fix the code while maintaining the original strategy logic. Here's the corrected version with Moon Dev themed improvements:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Load and preprocess data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names and structure
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VolatilitySurge(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    atr_period = 1920    # 20 days in 15m intervals
    sma_period = 19200   # 200 days in 15m intervals
    lookback_period = 35040  # 1 year in 15m intervals
    exit_bars = 1920     # 20 days in 15m intervals

    def init(self):
        # Calculate core indicators using TA-Lib
        self.atr20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.sma200 = self.I(talib.SMA, self.data.Close, self.sma_period)
        
        # Custom indicator for ATR percentile
        def calc_percentile(series):
            return series.rolling(self.lookback_period).quantile(0.10)
        self.atr_percentile = self.I(calc_percentile, self.atr20)

        # Trade tracking variables
        self.entry_price = None
        self.entry_atr = None
        self.highest_price = None
        self.bars_held = 0

    def next(self):
        # Moon Dev progress tracker
        if len(self.data) % 1000 == 0:
            print(f"🌙 Processing candle {len(self.data)} - Equity: ${self.equity:,.2f} ✨")

        if not self.position:
            # Entry logic
            if (len(self.atr20) > self.lookback_period and 
                not np.isnan(self.atr20[-1]) and
                not np.isnan(self.atr_percentile[-1]) and
                not np.isnan(self.sma200[-1])):
                
                if (self.atr20[-1] < self.atr_percentile[-1] and 
                    self.data.Close[-1] > self.sma200[-1]):
                    
                    print(f"\n🚀🌕 MOON SHOT DETECTED! 🚀")
                    print(f"ATR20({self.atr_period}): {self.atr20[-1]:.2f} < 10th %ile: {self.atr_percentile[-1]:.2f}")
                    print(f"Close ${self.data.Close[-1]:.2f} > SMA200 ${self.sma200[-1]:.2f}")

                    # Risk calculation
                    risk_amount = self.equity * self.risk_percent
                    stop_distance = 2 * self.atr20[-1]
                    position_size = int(round(risk_amount / stop_distance))

                    if position_size > 0:
                        self.buy(size=position_size)
                        self.entry_price = self.data.Close[-1]
                        self.entry_atr = self.atr20[-1]
                        self.highest_price = self.data.Close[-1]
                        self.bars_held = 0
                        print(f"📈 LONG ENTRY: {position_size} units at ${self.entry_price:.2f}")
                    else:
                        print("⚠️ Zero position size - risk parameters too tight!")
        else:
            # Update trailing values
            self.highest_price = max(self.highest_price, self.data.Close[-1])
            self.bars_held += 1

            # Exit conditions
            stop_price = self.entry_price - 2 * self.entry_atr
            trail_price = self.highest_price * 0.95

            if self.data.Close[-1] < stop_price or self.data.Close[-1] < trail_price or self.bars_held >= self.exit_bars:
                self.position.close()
                print(f"