Here's the debugged backtest code with Moon Dev themed improvements while maintaining the original strategy logic:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and prepare data 🌙
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
try:
    data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
except FileNotFoundError:
    raise FileNotFoundError("🌑 LUNAR ALERT! Data file not found. Check your moon path.")

# Clean column names and format
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class VolatilityBreakout(Strategy):
    def init(self):
        # 🌙✨ Core Indicators Calculation
        self.atr10 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=10, name='ATR10')
        self.atr10_ma20 = self.I(talib.SMA, self.atr10, timeperiod=20, name='ATR_MA20')
        self.donchian_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='DONCHIAN_HIGH')
        self.volume_ma50 = self.I(talib.SMA, self.data.Volume, timeperiod=50, name='VOLUME_MA50')
        
        # 🌕 Trailing Stop Tracking
        self.max_high = 0.0  # Explicit float initialization

    def next(self):
        # Wait for sufficient data
        if len(self.data.Close) < 50:
            return

        # 🌙✨ Entry Logic
        if not self.position:
            # Volatility contraction condition
            volatility_contraction = self.atr10[-1] < self.atr10_ma20[-1]
            
            # Price breakout condition
            price_breakout = self.data.Close[-1] > self.donchian_high[-1]
            
            # Volume confirmation
            volume_surge = self.data.Volume[-1] > 1.2 * self.volume_ma50[-1]

            if volatility_contraction and price_breakout and volume_surge:
                # 🚀 Risk Management Calculations
                risk_pct = 0.01  # 1% risk per trade
                risk_amount = self.equity * risk_pct
                atr_value = self.atr10[-1]
                
                # Position sizing calculation with proper rounding
                stop_loss = self.data.Close[-1] - atr_value
                risk_per_unit = self.data.Close[-1] - stop_loss
                position_size = risk_amount / risk_per_unit
                position_size = max(1, int(round(position_size)))  # Ensure minimum 1 unit

                if position_size > 0:
                    self.buy(size=position_size)
                    self.max_high = self.data.High[-1]  # Initialize trailing stop
                    print(f"🌕 MOON ALERT! Entry @ {self.data.Close[-1]:.2f}")
                    print(f"Size: {position_size} units | Risk: {risk_pct*100:.1f}% 🌙")
                    print("✨ Lunar Entry Conditions Met: Volatility Contraction + Breakout + Volume Surge ✨")

        # 🌑 Exit Logic
        elif self.position:
            # Update trailing stop
            self.max_high = max(self.max_high, self.data.High[-1])
            trailing_stop = self.max_high - 2 * self.atr10[-1]
            
            if self.data.Low[-1] <= trailing_stop:
                self.position.close()
                print(f"🌑 MOON EXIT! Price: {self.data.Close[-1]:.2f}")
                print(f"Profit: {self.position.pl_pct:.2f}% ✨")
                print("🚀 Trailing Stop Triggered - Returning to Lunar Orbit 🚀")

# Run backtest 🚀🌙
print("🌖🌗🌘🌒🌓🌔 MOON DEV BACKTEST INITIALIZING 🌔🌓🌒🌘🌗🌖")
bt = Backtest(data, VolatilityBreakout, cash=1_000_000, commission=.002)
stats = bt.run()
print("\n🌖🌗