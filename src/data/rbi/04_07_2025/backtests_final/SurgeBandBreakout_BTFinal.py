I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete corrected version with Moon Dev themed debug prints:

```python
# 🌙 Moon Dev SurgeBand Breakout Backtest ✨
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ====================
# 🛠 DATA PREPARATION
# ====================
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"

# Load and clean data
data = pd.read_csv(data_path)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Column mapping with proper case 🌙
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert datetime and set index 🕒
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

# ======================
# 🚀 SURGEBAND STRATEGY
# ======================
class SurgeBandBreakout(Strategy):
    bb_period = 20
    bb_dev = 2
    volume_ma_days = 20  # Days
    max_hold_hours = 48
    risk_pct = 0.02
    
    def init(self):
        # Convert days to 15m periods 🌙
        self.volume_ma_period = self.volume_ma_days * 24 * 4  # 15m intervals
        self.max_hold_bars = self.max_hold_hours * 4
        
        # 🌗 Bollinger Bands Calculation
        close = self.data.Close
        self.upper, self.middle, self.lower = self.I(
            lambda: talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev),
            name=['UPPER', 'MIDDLE', 'LOWER']
        )
        
        # 📈 Volume Surge Indicator
        self.volume_ma = self.I(talib.SMA, self.data.Volume, self.volume_ma_period)
        
        # 📆 Entry tracking for time exits
        self.entry_bar = 0

    def next(self):
        # Wait for indicators to warm up 🌙
        if len(self.data) < self.volume_ma_period:
            return
            
        # 🌊 Volume Surge Detection
        current_volume = self.data.Volume[-1]
        volume_threshold = self.volume_ma[-1] * 3  # 200% above average
        surge_detected = current_volume >= volume_threshold
        
        # 📉📈 Price and Bands
        price = self.data.Close[-1]
        upper = self.upper[-1]
        lower = self.lower[-1]
        
        # ==================
        # 🚪 ENTRY LOGIC
        # ==================
        if not self.position and surge_detected:
            # 🌕 Long Entry
            if price > upper:
                stop_price = self.lower[-1]
                risk_per_unit = price - stop_price
                if risk_per_unit > 0:
                    position_size = int(round((self.risk_pct * self.broker.equity) / risk_per_unit))
                    if position_size > 0:
                        self.buy(size=position_size, sl=stop_price)
                        self.entry_bar = len(self.data)
                        print(f"🚀🌙 MOON DEV LONG! Entry: {price:.2f} | Size: {position_size} | SL: {stop_price:.2f}")
            
            # 🌑 Short Entry
            elif price < lower:
                stop_price = self.upper[-1]
                risk_per_unit = stop_price - price
                if risk_per_unit > 0:
                    position_size = int(round((self.risk_pct * self.broker.equity) / risk_per_unit))
                    if position_size > 0:
                        self.sell(size=position_size, sl=stop_price)
                        self.entry_bar = len(self.data)
                        print(f"🌙🚀 MOON DEV SHORT! Entry: {price:.2f} | Size: {position_size} | SL: {stop_price:.2f}")

        # =================
        # 🚪 EXIT LOGIC
        # =================
        if self.position:
            # ⏳ Time-based Exit
            bars_held = len(self.data) - self.entry_bar
            if bars_held