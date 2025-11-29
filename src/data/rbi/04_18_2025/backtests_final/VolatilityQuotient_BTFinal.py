Here's the debugged backtest code with Moon Dev themed improvements while preserving the original strategy logic:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and prepare data with proper error handling
try:
    data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
    print("🌕 Data successfully loaded! Preparing for launch...")
except FileNotFoundError:
    raise FileNotFoundError("🌑 CRITICAL: Data file not found. Please check the file path.")

# Clean column names consistently
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Standardize column naming convention
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True, errors='ignore')  # Added errors='ignore' for safety

# Convert datetime with validation
try:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    print("⏱️  Time index successfully established. Countdown initiated!")
except Exception as e:
    raise ValueError(f"🌑 Time conversion failed: {str(e)}")

class VolatilityQuotient(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    stop_loss_pct = 0.02  # 2% stop loss
    time_exit_bars = 960  # 10 days in 15m intervals (96 bars/day)
    
    def init(self):
        # Calculate indicators with proper initialization
        self.returns = self.I(talib.ROCP, self.data.Close, timeperiod=1)
        self.hv_10 = self.I(talib.STDDEV, self.returns, timeperiod=10, nbdev=1)
        self.ma_hv_20 = self.I(talib.SMA, self.hv_10, timeperiod=20)
        self.rsi_5 = self.I(talib.RSI, self.data.Close, timeperiod=5)
        
        self.entry_bar = None
        print("🌖 Indicators initialized. Awaiting trading signals...")

    def next(self):
        current_close = self.data.Close[-1]
        
        # Enhanced Moon-themed debug prints
        if len(self.hv_10) > 20 and len(self.data) % 100 == 0:  # Print every 100 bars
            print(f"🌓 Volatility Scan | Current HV: {self.hv_10[-1]:.4f} | MA_HV: {self.ma_hv_20[-1]:.4f} | RSI: {self.rsi_5[-1]:.2f}")
        
        # Entry logic with proper position sizing
        if not self.position:
            if self.ma_hv_20[-2] > self.hv_10[-2] and self.ma_hv_20[-1] < self.hv_10[-1]:
                risk_amount = self.equity * self.risk_percent
                stop_loss_price = current_close * (1 - self.stop_loss_pct)
                risk_per_share = current_close - stop_loss_price
                
                if risk_per_share <= 0:
                    print("🌘 ABORT! Invalid risk calculation (stop loss >= entry)")
                    return
                
                position_size = int(round(risk_amount / risk_per_share))
                if position_size == 0:
                    print("🌘 ABORT! Position size too small (insufficient capital)")
                    return
                
                # Execute buy with proper position sizing
                self.buy(
                    size=position_size,  # Already rounded to whole number
                    sl=stop_loss_price,
                    tag="🌔 Volatility Breakout Entry"
                )
                self.entry_bar = len(self.data)
                print(f"🚀 LIFTOFF! Long {position_size} units at {current_close:.2f} | SL: {stop_loss_price:.2f}")

        # Exit logic with proper position closing
        else:
            # RSI exit condition
            if self.rsi_5[-2] < 70 and self.rsi_5[-1] > 70:
                self.position.close()
                print(f"🌒 RSI OVERBOUGHT! Exiting at {current_close:.2f}")
                self.entry_bar = None
                
            # Time-based exit
            elif self.entry_bar and (len(self.data) - self.entry_bar >=