Here's the fixed code with all backtesting.lib references removed and Moon Dev themed debug prints added:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and preprocess data
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean column names and drop unnamed columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename columns to match backtesting requirements
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

print("🌙✨ MOON DEV DATA PREP: Data loaded and preprocessed successfully ✨🚀")

class QuantileCompression(Strategy):
    bb_period = 20
    bb_dev = 2
    quantile_window = 90
    entry_quantile = 0.1
    exit_quantile = 0.5
    consecutive_bars = 3
    rsi_period = 14
    risk_pct = 0.01
    
    def init(self):
        # Bollinger Bands components
        self.mid = self.I(talib.SMA, self.data.Close, timeperiod=self.bb_period)
        self.std = self.I(talib.STDDEV, self.data.Close, timeperiod=self.bb_period)
        self.upper = self.I(lambda m, s: m + s*self.bb_dev, self.mid, self.std)
        self.lower = self.I(lambda m, s: m - s*self.bb_dev, self.mid, self.std)
        
        # Bollinger Bandwidth
        self.bbw = self.I(lambda u, l, m: (u-l)/m, self.upper, self.lower, self.mid)
        
        # Quantile calculations
        self.entry_threshold = self.I(
            lambda x: x.rolling(self.quantile_window).quantile(self.entry_quantile), 
            self.bbw, name='Entry Threshold'
        )
        self.exit_threshold = self.I(
            lambda x: x.rolling(self.quantile_window).quantile(self.exit_quantile),
            self.bbw, name='Exit Threshold'
        )
        
        # Additional indicators
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        print("🌙✨ MOON DEV INIT: Strategy initialized with stellar indicators ✨🚀")
        print("🌙📊 Indicators deployed:")
        print(f"   - Bollinger Bands ({self.bb_period} period)")
        print(f"   - RSI ({self.rsi_period} period)")
        print(f"   - ATR (14 period)")
        print(f"   - Quantile Compression ({self.quantile_window} window)")

    def next(self):
        if len(self.data) < max(self.quantile_window, self.bb_period) + 1:
            return
        
        # Entry logic
        compression_cluster = all(self.bbw[-i] < self.entry_threshold[-i] for i in range(1, self.consecutive_bars+1))
        price_near_mid = abs(self.data.Close[-1] - self.mid[-1])/self.mid[-1] < 0.01
        rsi_valid = 30 < self.rsi[-1] < 70
        
        # Exit logic
        volatility_spike = self.bbw[-1] > self.exit_threshold[-1]
        breakout = self.data.Close[-1] > self.upper[-1] or self.data.Close[-1] < self.lower[-1]
        
        # Trade execution
        if not self.position:
            if compression_cluster and price_near_mid and rsi_valid:
                sl = self.lower[-1]
                risk = self.data.Close[-1] - sl
                if risk <= 0: 
                    print("🌙⚠️ MOON DEV WARNING: Invalid risk calculation, skipping trade")
                    return
                
                position_size = int(round((self.equity * self.risk_pct) / risk))
                if position_size > 0:
                    self.buy(size=position_size, sl=