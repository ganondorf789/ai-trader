I'll fix the code by removing all `backtesting.lib` imports and replacing any usage of `crossover` or `crossunder` with manual implementations. Here's the corrected version with Moon Dev themed debug prints:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# Load and preprocess data
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map column names to backtesting.py requirements
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class FundingVwapRejection(Strategy):
    risk_percent = 0.01
    entry_funding = 0.001
    exit_funding = 0.0005
    emergency_funding = 0.003
    
    def init(self):
        # Weekly VWAP (672 periods = 7 days * 96 periods/day)
        def calc_vwap(high, low, close, volume):
            return ta.vwap(high=high, low=low, close=close, volume=volume, length=672)
        self.vwap = self.I(calc_vwap, self.data.High, self.data.Low, self.data.Close, self.data.Volume, name='VWAP')
        
        # Daily Bollinger Bands (20 days * 96 periods = 1920)
        def bb_upper(close): return talib.BBANDS(close, 1920, 2, 2)[0]
        def bb_lower(close): return talib.BBANDS(close, 1920, 2, 2)[2]
        self.bb_upper = self.I(bb_upper, self.data.Close, name='BB_Upper')
        self.bb_lower = self.I(bb_lower, self.data.Close, name='BB_Lower')
        
        # Swing high/low indicators
        self.swing_high = self.I(talib.MAX, self.data.High, 20, name='Swing_High')
        self.swing_low = self.I(talib.MIN, self.data.Low, 20, name='Swing_Low')
        
        # Volume SMA for validation
        self.vol_sma = self.I(talib.SMA, self.data.Volume, 20, name='Vol_SMA')
        
    def next(self):
        current_idx = len(self.data)-1
        print(f"\n🌙 Moon Dev Processing {self.data.index[-1]} ✨")
        
        if not self.position:
            # Funding rate check
            funding = self.data.df['funding_rate'].iloc[-1]
            if funding <= self.entry_funding:
                print(f"🌑 Funding rate {funding*100:.2f}% ≤ entry threshold (too low)", flush=True)
                return
                
            # VWAP cluster check (2+ touches in last 20 candles)
            touches = sum(1 for i in range(1,21) if (
                self.data.High[-i] >= self.vwap[-i] or
                self.data.Low[-i] <= self.vwap[-i]
            ))
            if touches < 2:
                print(f"🌓 Only {touches} VWAP touches (need ≥2)", flush=True)
                return
                
            # Rejection candle check
            curr_body = abs(self.data.Close[-1] - self.data.Open[-1])
            upper_wick = self.data.High[-1] - max(self.data.Close[-1], self.data.Open[-1])
            is_pinbar = upper_wick > curr_body * 1.5 and curr_body > 0
            
            engulf_cond = (
                self.data.Close[-1] < self.data.Open[-1] and  # Bearish candle
                self.data.Open[-1] > self.data.Close[-2] and  # Engulfs previous close
                self.data.Close[-1] < self.data.Open[-2]
            )
            
            if not (is_pinbar or engulf_cond):
                print("🌗 No valid rejection pattern detected", flush=True)
                return
                
            # Volume check
            if self.data.Volume[-1] < self.vol_sma[-1]:
                print(f"🌘