I'll fix the code by removing all `backtesting.lib` imports and replacing the crossover function with manual checks. Here's the corrected version with Moon Dev themed debug prints:

```python
import pandas as pd
import talib
from backtesting import Strategy, Backtest

# Load and preprocess data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename columns to match backtesting requirements
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
})

# Convert index to datetime
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VoltaBandExpansion(Strategy):
    bb_period = 20
    bb_dev = 2
    bbwp_threshold = 20
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    atr_period = 14
    risk_pct = 0.01  # 1% risk per trade
    
    def init(self):
        # Calculate Bollinger Bands components
        self.bb_upper = self.I(self._bb_upper, self.data.Close)
        self.bb_middle = self.I(self._bb_middle, self.data.Close)
        self.bb_lower = self.I(self._bb_lower, self.data.Close)
        
        # Calculate Bollinger Band Width
        bb_width = (self.bb_upper - self.bb_lower) / self.bb_middle
        self.bb_width = self.I(lambda: bb_width, name='BB_Width')
        
        # Calculate BBWP using RANK (percentile approximation)
        self.bbwp = self.I(talib.RANK, self.bb_width, timeperiod=20)
        self.bbwp = self.I(lambda x: 100 * (19 - x)/19, self.bbwp)  # Convert rank to percentile
        
        # Calculate MACD components
        self.macd_line = self.I(self._macd_line, self.data.Close)
        self.signal_line = self.I(self._signal_line, self.data.Close)
        
        # Calculate ATR for stop loss
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        print("🌙 MOON DEV INIT COMPLETE! Indicators ready for launch! 🚀")
        
    def _bb_upper(self, close):
        upper, _, _ = talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)
        return upper
        
    def _bb_middle(self, close):
        _, middle, _ = talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)
        return middle
        
    def _bb_lower(self, close):
        _, _, lower = talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)
        return lower
        
    def _macd_line(self, close):
        macd, _, _ = talib.MACD(close, self.macd_fast, self.macd_slow, self.macd_signal)
        return macd
        
    def _signal_line(self, close):
        _, signal, _ = talib.MACD(close, self.macd_fast, self.macd_slow, self.macd_signal)
        return signal
        
    def next(self):
        if len(self.data.Close) < 50:  # Ensure enough data
            return
            
        current_close = self.data.Close[-1]
        prev_close = self.data.Close[-2]
        bbwp = self.bbwp[-1]
        atr = self.atr[-1]
        
        # Long Entry Conditions (replaced crossover with manual check)
        long_trigger = (
            bbwp < self.bbwp_threshold and
            (self.macd_line[-2] < self.signal_line[-2] and self.macd_line[-1] > self.signal_line[-1]) and  # Bullish crossover
            current_close > self.bb_upper[-1] and
            prev_close <= self.bb_upper[-2]
        )
        
        # Short Entry Conditions (replaced crossover with manual check