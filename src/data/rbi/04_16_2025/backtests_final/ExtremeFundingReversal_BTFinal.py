Here's the debugged code with all technical issues fixed while preserving the original strategy logic:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# DATA PREPROCESSING 🌙
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'])

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data.set_index('datetime', inplace=True)
assert 'funding_rate' in data.columns, "🌑 CRITICAL: Missing funding_rate column!"

class ExtremeFundingReversal(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # Funding Rate Indicators 📈
        fr = self.data.df['funding_rate']
        self.fund_mean = self.I(talib.SMA, fr, timeperiod=2880)  # 30-day mean
        self.fund_std = self.I(talib.STDDEV, fr, timeperiod=2880)  # 30-day std
        self.upper_band = self.I(lambda: self.fund_mean + 2*self.fund_std)
        self.lower_band = self.I(lambda: self.fund_mean - 2*self.fund_std)
        
        # Volume & Volatility Indicators 🌊
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=1344)  # 14-day MA
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, 
                         self.data.Close, timeperiod=14)  # 14-period ATR
        
    def next(self):
        # Current values ✨
        price = self.data.Close[-1]
        fr = self.data.df['funding_rate'][-1]
        vol = self.data.Volume[-1]
        
        if self.position:
            # Exit Conditions 🚪
            current_upper_1std = self.fund_mean[-1] + self.fund_std[-1]
            current_lower_1std = self.fund_mean[-1] - self.fund_std[-1]
            
            # Primary exit: Funding normalized 🌙
            if (self.position.is_short and fr <= current_upper_1std) or \
               (self.position.is_long and fr >= current_lower_1std):
                self.position.close()
                print(f"🌙✨ Funding normalized! Exiting at {price:.2f}")
            
            # Secondary exit: ATR stop 🛑
            elif self.position.is_short and price > self.entry_price + 1.5*self.entry_atr:
                self.position.close()
                print(f"🚨🌑 SHORT Stop triggered! {price:.2f} > {self.entry_price + 1.5*self.entry_atr:.2f}")
            elif self.position.is_long and price < self.entry_price - 1.5*self.entry_atr:
                self.position.close()
                print(f"🚨🌑 LONG Stop triggered! {price:.2f} < {self.entry_price - 1.5*self.entry_atr:.2f}")
                
        else:
            # Entry Conditions 🚀
            vol_spike = vol > 1.2 * self.vol_ma[-1]
            atr_val = self.atr[-1] if self.atr[-1] > 0 else 1
            
            if fr > self.upper_band[-1] and vol_spike:
                # Calculate position size 🌙
                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / (1.5 * atr_val)))
                self.entry_price = price
                self.entry_atr = atr_val
                self.sell(size=size)
                print(f"🌙🚀 SHORT! Funding {fr:.6f} > {self.upper_band[-1]:.6f}")
                print(f"   Size: {size}, Risk: {1.5*atr_val:.2f} ATR")
                
            elif fr < self.lower_band[-1] and vol_spike:
                risk