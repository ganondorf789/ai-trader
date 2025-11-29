```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

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
        self.mid = self.I(talib.SMA, self.data.Close, self.bb_period)
        self.std = self.I(talib.STDDEV, self.data.Close, self.bb_period)
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
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
        print("🌙✨ MOON DEV INIT: Strategy initialized with stellar indicators ✨🚀")

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
                if risk <= 0: return
                
                position_size = int(round((self.equity * self.risk_pct) / risk))
                if position_size > 0:
                    self.buy(size=position_size, sl=sl, tp=self.data.Close[-1]+1.5*self.atr[-1])
                    print(f"🌙🚀 MOON DEV ENTRY: Long {position_size} units at {self.data.Close[-1]} ✨")
        
        else:
            if volatility_spike or breakout:
                self.position.close()
                print(f"🌙🌑 MOON DEV EXIT: Closing position at {self.data.Close[-1]} ✨")

# Run backtest
bt = Backtest(data, QuantileCompression, cash=1_000_000, commission=.002)
stats = bt.run()
print("\n🌕🌖🌗🌘 MOON DEV BACK