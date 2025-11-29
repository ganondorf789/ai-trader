I'll fix the code while maintaining the original strategy logic. Here are the key issues I found and fixed:

1. Fixed incomplete print statement (missing closing brace)
2. Ensured position sizing is properly rounded to whole numbers
3. Added proper string formatting for debug prints
4. Fixed variable naming consistency
5. Added missing imports
6. Ensured all calculations use proper price levels (not distances)

Here's the complete fixed code:

```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# Data preparation
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VoltaicBreakout(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # Chaikin Volatility Calculation ✨
        def chaikin_vol(high, low, window):
            chv = ta.chaikin_volatility(high=high, low=low, window=window)
            return chv[f'CHV_{window}']
        self.chai_vol = self.I(chaikin_vol, self.data.High, self.data.Low, 10)
        
        # 24-hour SMA (96 periods for 15m data) 🚀
        self.chai_vol_sma = self.I(talib.SMA, self.chai_vol, 96)
        
        # Supertrend with 10-period ATR and 3x multiplier 🌊
        def supertrend_dir(high, low, close, length, multiplier):
            st = ta.supertrend(high=high, low=low, close=close, 
                              length=length, multiplier=multiplier)
            return st[f'SUPERTd_{length}_{multiplier}']
        self.supertrend = self.I(supertrend_dir, self.data.High, self.data.Low, 
                                self.data.Close, 10, 3)
        
        # Volatility measurements 💎
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, 
                         self.data.Close, 14)
        self.atr_sma = self.I(talib.SMA, self.atr, 20)
        
        self.trailing_high = None  # Track highest high since entry 🌙

    def next(self):
        # Moon Dev debug prints ✨
        if len(self.data) % 100 == 0:
            print(f"🌙 Processing bar {len(self.data)} | Close: {self.data.Close[-1]:.2f}")
        
        # Skip initial bars ❌
        if len(self.data) < 96 + 3 or len(self.supertrend) < 3:
            return
        
        # Entry Logic 🚀
        if not self.position:
            # Chaikin Volatility crossover check (manual implementation) 🌙
            vol_cross = (self.chai_vol[-3] < self.chai_vol_sma[-3] and 
                         self.chai_vol[-2] > self.chai_vol_sma[-2])
            
            # Supertrend bullish flip check ✅
            st_current = self.supertrend[-2]
            st_previous = self.supertrend[-3]
            st_flip = st_current == 1 and st_previous != 1
            
            if vol_cross and st_flip:
                # Volatility filter check 🌊
                atr_val = self.atr[-2]
                atr_sma_val = self.atr_sma[-2]
                
                if atr_val > 1.5 * atr_sma_val:
                    # Position sizing calculation 💰
                    entry_price = self.data.Open[-1]
                    stop_loss = entry_price - 2 * atr_val
                    risk_per_share = entry_price - stop_loss
                    
                    if risk_per_share > 0:
                        equity_risk = self.equity * self.risk_percent
                        position_size = int(round(equity_risk / risk_per_share))
                        position_size = min(position_size, int(self.equity // entry