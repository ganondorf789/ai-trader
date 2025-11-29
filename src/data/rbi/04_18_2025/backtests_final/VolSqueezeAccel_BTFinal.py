I'll fix the incomplete code while maintaining the original strategy logic. Here's the debugged version with Moon Dev themed improvements:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and preprocess data
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

# Clean column names and drop unnamed columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

# Rename columns to match backtesting requirements
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert datetime and set index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolSqueezeAccel(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    max_drawdown = 0.05    # 5% max portfolio drawdown
    
    def init(self):
        # Bollinger Bands components
        def bband_upper(close):
            upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return upper
        self.bb_upper = self.I(bband_upper, self.data.Close)
        
        def bband_middle(close):
            _, middle, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return middle
        self.bb_middle = self.I(bband_middle, self.data.Close)
        
        def bband_lower(close):
            _, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return lower
        self.bb_lower = self.I(bband_lower, self.data.Close)

        # Bollinger Band Width calculations
        self.bb_width = self.I(lambda u, l, m: (u - l) / m, 
                              self.bb_upper, self.bb_lower, self.bb_middle)
        
        # 6-month rolling window (17280 periods for 15m data)
        self.bb_width_min = self.I(talib.MIN, self.bb_width, 17280)
        self.bb_width_max = self.I(talib.MAX, self.bb_width, 17280)
        self.bb_width_pct = self.I(lambda w, mn, mx: (w - mn) / (mx - mn + 1e-9),
                                  self.bb_width, self.bb_width_min, self.bb_width_max)

        # Volume indicators
        self.vol_sma = self.I(talib.SMA, self.data.Volume, 1920)  # 20-day SMA
        
        # VIX term structure ratio
        self.vix_ratio = self.I(lambda x: x['vix_front']/x['vix_3m'], self.data.df)
        
        self.trades = []
        print("🌙✨ Moon Dev Indicators Initialized! Ready for cosmic trading!")

    def next(self):
        current_close = self.data.Close[-1]
        equity = self.equity

        # 🌙✨ Moon Dev Risk Check
        if equity < 1_000_000 * (1 - self.max_drawdown):
            print("🌙⚠️ EMERGENCY! Portfolio drawdown limit reached! Aborting mission!")
            self.position.close()
            return

        if not self.position:
            # Entry conditions check
            bb_squeeze = self.bb_width_pct[-1] <= 0.2
            vol_spike = self.data.Volume[-1] > 1.5 * self.vol_sma[-1]
            vix_confirm = self.vix_ratio[-1] < 1.15
            
            # Long setup
            if (bb_squeeze and vol_spike and vix_confirm and
                self.data.High[-2] < self.bb_upper[-2] and self.data.High[-1] > self.bb_upper[-1]):
                
                sl = self.bb_lower[-1]
                risk_per_share = current_close - sl
                if risk_per_share > 0:
                    size = int(round((equity * self.risk_per_trade) / risk_per_share))
                    if size >