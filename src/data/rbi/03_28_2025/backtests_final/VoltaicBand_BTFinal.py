I'll fix the code while maintaining the original strategy logic. Here's the debugged version with Moon Dev themed prints:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import numpy as np

# Data preparation
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class VoltaicBand(Strategy):
    risk_percent = 0.01
    atr_period = 14
    vwap_period = 20
    
    def init(self):
        # Calculate VWAP
        typical_price = (self.data.High + self.data.Low + self.data.Close) / 3
        vwap_num = typical_price * self.data.Volume
        vwap_den = self.data.Volume
        
        self.vwap = self.I(lambda: (
            vwap_num.rolling(self.vwap_period).sum() / 
            vwap_den.rolling(self.vwap_period).sum()
        ).values, name='VWAP')
        
        # Calculate ATR
        self.atr = self.I(talib.ATR, 
                         self.data.High, 
                         self.data.Low, 
                         self.data.Close, 
                         self.atr_period,
                         name='ATR')
        
        self.entry_vwap = None
        self.tracking_high = None
        self.tracking_low = None

    def next(self):
        # Wait for indicators to warm up
        if len(self.data) < self.vwap_period or len(self.atr) < self.atr_period:
            return
        
        current_close = self.data.Close[-1]
        prev_close = self.data.Close[-2]
        
        # Get previous bar's indicator values
        prev_vwap = self.vwap[-2]
        prev_atr = self.atr[-2]
        upper_band_prev = prev_vwap + 2 * prev_atr
        lower_band_prev = prev_vwap - 2 * prev_atr
        
        # Moon Dev debug logging
        print(f"🌙 Bar: {len(self.data)} | Close: {current_close:.2f} | VWAP: {self.vwap[-1]:.2f} | ATR: {self.atr[-1]:.2f}")
        
        # Entry logic
        if not self.position:
            # Long entry - bullish crossover check
            if prev_close <= upper_band_prev and current_close > upper_band_prev:
                risk_amount = self.equity * self.risk_percent
                entry_price = current_close
                stop_loss = self.vwap[-1]  # Current bar's VWAP
                risk_per_share = entry_price - stop_loss
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.buy(size=position_size)
                        self.entry_vwap = stop_loss
                        self.tracking_high = entry_price
                        print(f"🚀🌕 MOON SHOT! Long {position_size} units at {entry_price:.2f}")
            
            # Short entry - bearish crossover check
            elif prev_close >= lower_band_prev and current_close < lower_band_prev:
                risk_amount = self.equity * self.risk_percent
                entry_price = current_close
                stop_loss = self.vwap[-1]  # Current bar's VWAP
                risk_per_share = stop_loss - entry_price
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.sell(size=position_size)
                        self.entry_vwap = stop_loss
                        self.tracking_low = entry_price
                        print(f"🔻🌑 BLACK HOLE! Short {position_size} units at {entry_price:.2f}")
        
        # Exit logic
        else:
            if self.position.is_long:
                self.tracking_high = max(self.tracking_high, self.data.High[-1])
                # Calculate 50% retracement level
                exit