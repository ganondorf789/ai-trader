Here's the debugged code with Moon Dev themed improvements while maintaining the original strategy logic:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Load and preprocess data with Moon Dev precision 🌙
try:
    data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
    print("🌙 MOON DEV DATA LOADED SUCCESSFULLY! ✨")
except FileNotFoundError:
    raise FileNotFoundError("🌑 MOON DEV ERROR: Data file not found. Please check the path.")

# Clean column names and drop unnamed columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename columns to match backtesting.py requirements with celestial precision 🌕
required_columns = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'open_interest': 'OpenInterest'
}

# Check for missing columns
missing_cols = [col for col in required_columns.values() if col not in data.columns]
if missing_cols:
    print(f"🌒 MOON DEV WARNING: Missing columns {missing_cols} - strategy may not work as intended")

data.rename(columns=required_columns, inplace=True)

class VolSqueezeSurge(Strategy):
    risk_percent = 0.01  # 1% risk per trade - strictly maintained 🌙
    
    def init(self):
        # Calculate indicators with lunar precision 🌙✨
        self.middle_band = self.I(talib.SMA, self.data.Close, timeperiod=20)
        self.stddev = self.I(talib.STDDEV, self.data.Close, timeperiod=20)
        
        # Ensure OpenInterest exists before calculating
        if 'OpenInterest' in self.data.df.columns:
            self.oi_ma = self.I(talib.SMA, self.data.OpenInterest, timeperiod=960)
        else:
            self.oi_ma = self.I(lambda x: np.zeros(len(x)), self.data.Close)
            print("🌓 MOON DEV WARNING: OpenInterest not found - using zeros")
            
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        # Track Bollinger Width history with stellar accuracy ✨
        self.bb_widths = []
        
    def next(self):
        # Wait for sufficient data - moon phases require patience 🌑🌕
        if len(self.data) < 960 or len(self.bb_widths) < 100:
            return
        
        # Calculate current Bollinger values
        upper = self.middle_band[-1] + 2 * self.stddev[-1]
        lower = self.middle_band[-1] - 2 * self.stddev[-1]
        current_bb_width = upper - lower
        self.bb_widths.append(current_bb_width)
        self.bb_widths = self.bb_widths[-100:]  # Keep rolling 100 periods
        
        # Calculate percentiles with cosmic precision 🌌
        bb_p10 = np.percentile(self.bb_widths, 10)
        bb_p50 = np.percentile(self.bb_widths, 50)
        current_close = self.data.Close[-1]
        
        if not self.position:
            # Entry conditions - lunar alignment check 🌙✨
            oi_ma_value = self.oi_ma[-1]
            current_oi = self.data.OpenInterest[-1] if 'OpenInterest' in self.data.df.columns else 0
            
            if (current_bb_width < bb_p10 and
                oi_ma_value > 0 and
                (current_oi - oi_ma_value)/oi_ma_value > 0.2 and
                lower < current_close < upper):
                
                # Risk management - celestial safeguards 🚀
                atr_value = self.atr[-1]
                stop_loss = current_close - 2 * atr_value
                risk_amount = self.equity * self.risk_percent
                position_size = int(round(risk_amount / (current_close - stop_loss)))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_loss)
                    print(f"🌙 MOON DEV ENTRY 🚀 | Price: {current_close:.2f} | Size: {position_size} | SL: {stop_loss:.2f}")