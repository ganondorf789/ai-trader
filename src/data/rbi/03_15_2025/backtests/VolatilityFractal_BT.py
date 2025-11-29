Here's the implementation of the VolatilityFractal strategy for backtesting.py:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Data Preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and format data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'datetime': 'Date'
}, inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

class VolatilityFractal(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    atr_multiplier = 1.5
    rr_ratio = 2
    
    def init(self):
        # Core Indicators
        self.rsi = self.I(talib.RSI, self.data.Close, 14)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
        # MACD components
        macd, macd_signal, _ = talib.MACD(self.data.Close, 12, 26, 9)
        self.macd_line = self.I(lambda: macd, name='MACD')
        self.macd_signal = self.I(lambda: macd_signal, name='MACD Signal')
        
        # Swing detection
        self.swing_high = self.I(talib.MAX, self.data.High, 5)
        self.swing_low = self.I(talib.MIN, self.data.Low, 5)
        self.volume_peak = self.I(talib.MAX, self.data.Volume, 5)
        
        # Track market state
        self.price_peaks = []
        self.price_troughs = []
        self.rsi_peaks = []
        self.rsi_troughs = []

    def next(self):
        # Moon Dev Debug Prints 🌙
        print(f"\n🌕 New Candle: {self.data.index[-1]} | Close: {self.data.Close[-1]} 🌑")
        
        # Update swing points
        self._update_swings()
        
        # Check entries
        if not self.position:
            self._check_entries()
            
        # Manage exits
        if self.position:
            self._manage_exits()

    def _update_swings(self):
        # Price swings
        if self.data.High[-1] == self.swing_high[-1]:
            self.price_peaks.append((len(self.data)-1, self.data.High[-1]))
            if len(self.price_peaks) > 3:
                self.price_peaks.pop(0)
            print(f"✨ New Price Peak: {self.data.High[-1]} ✨")

        if self.data.Low[-1] == self.swing_low[-1]:
            self.price_troughs.append((len(self.data)-1, self.data.Low[-1]))
            if len(self.price_troughs) > 3:
                self.price_troughs.pop(0)
            print(f"🌑 New Price Trough: {self.data.Low[-1]} 🌑")

        # RSI swings
        if self.rsi[-1] == self.I(talib.MAX, self.rsi, 5)[-1]:
            self.rsi_peaks.append((len(self.data)-1, self.rsi[-1]))
            if len(self.rsi_peaks) > 3:
                self.rsi_peaks.pop(0)
            print(f"📈 New RSI Peak: {self.rsi[-1]} 📈")

        if self.rsi[-1] == self.I(talib.MIN, self.rsi, 5)[-1]:
            self.rsi_troughs.append((len(self.data)-1, self.rsi[-1]))
            if len(self.rsi_troughs) > 3:
                self.rsi_troughs.pop(0)
            print(f"📉 New RSI Trough: {self.rsi[-1]} 📉")

    def _check_entries(self):
        # Bullish divergence check
        bull_div = len(self.price_troughs) >= 2 and len(self.rsi_troughs) >= 2
        if bull_div:
            price_up = self.price_trough