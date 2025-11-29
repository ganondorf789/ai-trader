I'll fix the incomplete code while maintaining the original strategy logic. Here's the debugged version with Moon Dev themed prints:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import numpy as np

# Load and prepare data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Format columns to backtesting.py requirements
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'datetime': 'DateTime'
})

# Convert to datetime index
data['DateTime'] = pd.to_datetime(data['DateTime'])
data = data.set_index('DateTime')

class LiquidityMomentum(Strategy):
    cmo_period = 14
    std_dev_period = 50
    std_multiplier = 1.5
    swing_window = 20
    fib_window = 100
    risk_pct = 0.01
    atr_period = 14
    
    def init(self):
        # Core momentum indicators 🌙
        self.cmo = self.I(talib.CMO, self.data.Close, self.cmo_period, name='CMO')
        std_dev = self.I(talib.STDDEV, self.cmo, self.std_dev_period, nbdev=1, name='STDDEV')
        self.upper_band = self.I(lambda: self.cmo + std_dev*self.std_multiplier, name='UPPER_BAND')
        self.lower_band = self.I(lambda: self.cmo - std_dev*self.std_multiplier, name='LOWER_BAND')
        
        # Liquidity zones ✨
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_window, name='SWING_HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_window, name='SWING_LOW')
        
        # Fibonacci levels 📐
        self.fib_high = self.I(talib.MAX, self.data.High, self.fib_window, name='FIB_HIGH')
        self.fib_low = self.I(talib.MIN, self.data.Low, self.fib_window, name='FIB_LOW')
        
        # Volatility measure 🌊
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR')
        
        print("🌙✨ Moon Dev indicators initialized! Ready for cosmic gains! 🚀")

    def next(self):
        price = self.data.Close[-1]
        fib_range = self.fib_high[-1] - self.fib_low[-1]
        fib_levels = {
            '38.2%': self.fib_low[-1] + fib_range*0.382,
            '61.8%': self.fib_low[-1] + fib_range*0.618
        }
        
        # Entry conditions 🌙🚀
        if not self.position:
            # Long entry cosmic alignment ✨
            if ((self.cmo[-3] < self.upper_band[-3] and self.cmo[-1] > self.upper_band[-1]) and
                any(abs(price - lvl) < price*0.01 for lvl in fib_levels.values()) and
                abs(price - self.swing_low[-1]) < price*0.01):
                
                risk_amount = self.equity * self.risk_pct
                sl = self.swing_low[-1] * 0.995
                position_size = int(round(risk_amount / (price - sl)))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=sl, 
                            tag=f"🌙 LONG | Size: {position_size} | SL: {sl:.1f}")
                    print(f"🚀🌕 BLAST OFF! Long entry at {price:.2f} ✨")

            # Short entry lunar eclipse 🌑
            elif ((self.cmo[-3] > self.lower_band[-3] and self.cmo[-1] < self.lower_band[-1]) and
                any(abs(price - lvl) < price*0.01 for lvl in fib_levels.values()) and
                abs(price -