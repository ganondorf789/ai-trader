Here's the fixed and completed backtest code with Moon Dev themed debug prints and proper position sizing implementation:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and preprocess data
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean data columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

# Rename columns to match backtesting.py requirements
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Validate required columns
assert all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']), "Missing OHLCV columns"
assert 'funding_rate' in data.columns, "Funding rate column missing"

class VolatilitySqueezeDivergence(Strategy):
    ema_period = 20
    atr_period_keltner = 20
    risk_pct = 0.01  # 1% risk per trade
    daily_loss_limit = 0.05  # 5% daily loss limit

    def init(self):
        # Moon Dev initialization message
        print("🌙 Initializing VolatilitySqueezeDivergence Strategy...")
        
        # Keltner Channel indicators
        self.ema = self.I(talib.EMA, self.data.Close, self.ema_period)
        self.atr_keltner = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period_keltner)
        self.upper_keltner = self.I(lambda: self.ema + 2*self.atr_keltner)
        self.lower_keltner = self.I(lambda: self.ema - 2*self.atr_keltner)
        self.keltner_width = self.I(lambda: self.upper_keltner - self.lower_keltner)
        self.avg_width = self.I(talib.SMA, self.keltner_width, self.ema_period)

        # Volume indicators
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20)

        # Funding rate indicators
        self.funding = self.I(lambda: self.data.df['funding_rate'].values)
        self.funding_baseline = self.I(talib.SMA, self.funding, 4)

        # Daily loss tracking
        self.prev_day = None
        self.daily_equity = self._broker.starting_cash
        self.daily_trading_halted = False

        print("✨ Indicators initialized: Keltner Channels, Volume MA, Funding Baseline")

    def next(self):
        # Check daily reset
        current_day = self.data.index[-1].date()
        if current_day != self.prev_day:
            self.daily_equity = self.equity
            self.prev_day = current_day
            self.daily_trading_halted = False
            print(f"🌙 New day {current_day} | Starting equity: ${self.daily_equity:,.2f}")

        # Check daily loss limit
        if self.daily_trading_halted:
            return
            
        daily_loss = (self.daily_equity - self.equity) / self.daily_equity
        if daily_loss >= self.daily_loss_limit:
            print(f"🚨 Daily loss {daily_loss:.2%}! Closing all trades!")
            for trade in self.trades:
                trade.close()
            self.daily_trading_halted = True
            return

        # Skip if in position
        if self.position:
            return

        # Calculate current values
        current_close = self.data.Close[-1]
        current_upper = self.upper_keltner[-1]
        current_lower = self.lower_keltner[-1]
        width_ratio = self.keltner_width[-1] / self.avg_width[-1]
        funding_diff = self.funding[-1] - self.funding_baseline[-1]

        # Long entry conditions
        if (width_ratio <= 0.5 and
            self.data.Volume[-1] > self.volume_ma[-1] and
            funding_diff >= 0.0001 and
            current_close > current_upper):
            
            # Position sizing with proper risk management
            stop_loss = min(self