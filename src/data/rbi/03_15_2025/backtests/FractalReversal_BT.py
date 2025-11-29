To address the problem, we'll create a backtesting implementation for the FractalReversal strategy using the specified requirements. Here's the structured solution:

```python
# backtest_fractal_reversal.py
import pandas as pd
from backtesting import Backtest, Strategy
import talib

# Load and preprocess data
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)

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

class FractalReversal(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    rr_ratio = 2     # Risk-reward ratio
    
    def init(self):
        # Moon Dev Indicator Initialization 🌙
        self.sma50 = self.I(talib.SMA, self.data.Close, 50, name='SMA 50')
        self.sma200 = self.I(talib.SMA, self.data.Close, 200, name='SMA 200')
        self.max_high = self.I(talib.MAX, self.data.High, 5, name='MAX High')
        self.min_low = self.I(talib.MIN, self.data.Low, 5, name='MIN Low')
        
        self.bullish_signal = False
        self.bearish_signal = False

    def next(self):
        # Moon Dev Crossover Detection 🌗
        if len(self.sma50) > 1 and len(self.sma200) > 1:
            # Detect MA crossovers
            ma_bullish = self.sma50[-2] < self.sma200[-2] and self.sma50[-1] > self.sma200[-1]
            ma_bearish = self.sma50[-2] > self.sma200[-2] and self.sma50[-1] < self.sma200[-1]
            
            if ma_bullish:
                print(f"🌕 BULLISH CROSSOVER detected! SMA50 ({self.sma50[-1]:.2f}) > SMA200 ({self.sma200[-1]:.2f})")
                self.bullish_signal = True
                self.bearish_signal = False
                
            if ma_bearish:
                print(f"🌑 BEARISH CROSSOVER detected! SMA50 ({self.sma50[-1]:.2f}) < SMA200 ({self.sma200[-1]:.2f})")
                self.bearish_signal = True
                self.bullish_signal = False

        # Moon Dev Fractal Detection 🔍
        fractal_bullish = False
        fractal_bearish = False
        
        if len(self.data.High) >= 5:
            fractal_bullish = self.data.High[-3] == self.max_high[-1]
            
        if len(self.data.Low) >= 5:
            fractal_bearish = self.data.Low[-3] == self.min_low[-1]

        # Moon Dev Trade Execution 💰
        if not self.position:
            # Bearish reversal entry (Short)
            if self.bearish_signal and fractal_bullish:
                print(f"🚨 BEARISH REVERSAL SIGNAL @ {self.data.index[-1]}")
                entry = self.data.Close[-1]
                sl = self.data.High[-3] * 1.005  # 0.5% above fractal high
                risk = sl - entry
                
                if risk > 0:
                    tp = entry - (risk * self.rr_ratio)
                    size = int(round((self.equity * self.risk_pct) / risk))
                    self.sell(size=size, sl=sl, tp=tp)
                    print(f"🚀 SHORT ENTRY {size} units | Entry: {entry:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
                    self.bearish_signal = False

            # Bullish reversal entry (Long)
            elif self.bullish_signal and fractal_bearish:
                print(f"🚨 BULLISH REVERSAL SIGNAL @ {self.data.index[-1]}")
                entry = self.data.Close[-1]
                sl = self