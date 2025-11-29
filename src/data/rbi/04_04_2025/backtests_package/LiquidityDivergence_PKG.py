Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations, along with Moon Dev themed debug prints:

```python
# 🌙 Moon Dev's Liquidity Divergence Backtest 🌙
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np

# Clean and prepare data
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class LiquidityDivergence(Strategy):
    swing_period = 20
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    atr_period = 14
    risk_pct = 0.01
    rr_ratio = 2

    def init(self):
        # Liquidity Zones 🌊
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period, name='LIQ HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period, name='LIQ LOW')
        
        # Divergence Engine 💹
        close = self.data.Close
        macd, macd_signal, macd_hist = talib.MACD(close, 
                                                 self.macd_fast,
                                                 self.macd_slow,
                                                 self.macd_signal)
        self.macd_hist = self.I(lambda: macd_hist, name='MACD Hist')
        
        # Risk Management Shield 🛡️
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        self.last_swing_high_val = None
        self.last_swing_low_val = None
        self.last_macd_peak = None
        self.last_macd_trough = None

    def calculate_position(self, direction, entry_price, stop_loss):
        """✨ Moon Dev Position Sizing Algorithm ✨"""
        risk_amount = self.equity * self.risk_pct
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == 0:
            print("🌘 WARNING: Zero risk detected! Skipping trade...")
            return
            
        units = risk_amount / risk_per_unit
        take_profit = entry_price + self.rr_ratio * (entry_price - stop_loss) if direction == 'long' else entry_price - self.rr_ratio * (stop_loss - entry_price)
        
        if direction == 'long':
            self.buy(size=units, sl=stop_loss, tp=take_profit)
        else:
            self.sell(size=units, sl=stop_loss, tp=take_profit)

    def next(self):
        if len(self.data) < 50:  # Warmup period
            return
            
        current_close = self.data.Close[-1]
        current_swing_high = self.swing_high[-1]
        current_swing_low = self.swing_low[-1]
        current_macd_hist = self.macd_hist[-1]
        current_atr = self.atr[-1] or 1  # Fallback to 1 if undefined
        
        # 🌙 Swing High Detection
        if current_swing_high != self.swing_high[-2]:
            self.last_swing_high_val = current_swing_high
            self.last_macd_peak = self.macd_hist[-1]
            print(f"🌕 NEW LIQUIDITY ZONE! Swing High {current_swing_high:.2f} | MACD Peak {self.last_macd_peak:.2f}")

        # 🌑 Swing Low Detection
        if current_swing_low != self.swing_low[-2]:
            self.last_swing_low_val = current_swing_low
            self.last_macd_trough = self.macd_hist[-1]
            print(f"🌑 NEW LIQUIDITY VOID! Swing Low {current_swing_low:.2f} | MACD Trough {