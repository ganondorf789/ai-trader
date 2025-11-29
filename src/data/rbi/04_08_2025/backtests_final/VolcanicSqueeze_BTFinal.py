Here's the fixed backtest code with Moon Dev themed debug prints and proper technical implementations:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Load and preprocess data with Moon Dev cosmic data validation 🌌
print("🌙✨ MOON DEV INITIALIZING COSMIC DATA DOWNLOAD...")
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names with lunar precision
print("🧹 CLEANSING DATA WITH MOON DUST FILTER...")
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col], errors='ignore')

# Standardize column names
column_mapping = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}
data.rename(columns=column_mapping, inplace=True)

# Set datetime index with cosmic alignment
print("⏰ SYNCHRONIZING MOON PHASE TIMESTAMPS...")
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolcanicSqueeze(Strategy):
    ema_period = 20
    atr_period = 14
    keltner_multiplier = 2
    volume_multiplier = 1.5
    risk_pct = 0.01
    fib_levels = [1.618, 2.618]
    trailing_atr_mult = 1.5

    def init(self):
        # Core indicators with lunar precision calibration
        print("🔭 ACTIVATING MOON-BASED TECHNICAL INDICATORS...")
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=self.ema_period)
        
        # Check if liquidationheatmap exists in data
        if 'liquidationheatmap' in self.data.df.columns:
            self.heatmap_sma = self.I(talib.SMA, self.data['liquidationheatmap'], timeperiod=self.ema_period)
        else:
            print("⚠️ MOON WARNING: Liquidation heatmap data not found - proceeding without")
            self.heatmap_sma = self.I(np.zeros, len(self.data.Close))
        
        # Swing extremes
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.ema_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.ema_period)

    def next(self):
        if self.position:
            self.manage_open_trades()
        else:
            self.check_entries()

    def check_entries(self):
        current_close = self.data.Close[-1]
        upper_band = self.ema[-1] + self.keltner_multiplier * self.atr[-1]
        lower_band = self.ema[-1] - self.keltner_multiplier * self.atr[-1]

        # Volume and heatmap validation
        vol_ok = self.data.Volume[-1] > self.volume_multiplier * self.vol_sma[-1]
        heat_ok = True  # Default to True if no heatmap data
        
        if 'liquidationheatmap' in self.data.df.columns:
            heat_ok = self.data['liquidationheatmap'][-1] > self.volume_multiplier * self.heatmap_sma[-1]

        if current_close > upper_band and vol_ok and heat_ok:
            self.enter_long(upper_band)
        elif current_close < lower_band and vol_ok and heat_ok:
            self.enter_short(lower_band)

    def enter_long(self, upper_band):
        stop_price = self.ema[-1] - self.keltner_multiplier * self.atr[-1]
        risk_distance = abs(self.data.Close[-1] - stop_price)
        position_size = self.calculate_size(risk_distance)
        
        if position_size > 0:
            print(f"🌕🚀 MOONSHOT ALERT! LONG ERUPTION DETECTED! Entry: {self.data.Close[-1]: