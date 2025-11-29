To address the requirements, here's the complete implementation of the VortexRetrace strategy for backtesting.py:

```python
# -*- coding: utf-8 -*-
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib
import pandas_ta as ta

# Data preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean and format data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VortexRetrace(Strategy):
    vi_period = 14
    swing_period = 20
    volume_ma_period = 5
    risk_pct = 0.01
    fib_levels = [0.382, 0.5, 0.618]
    
    def init(self):
        # Vortex Indicator using pandas_ta
        self.vi_plus = self.I(self._calculate_vortex, name='VI+', which='plus')
        self.vi_minus = self.I(self._calculate_vortex, name='VI-', which='minus')
        
        # Volume analysis
        self.volume_ma = self.I(talib.SMA, self.data.Volume, self.volume_ma_period)
        
        # Swing levels
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period)
        
        # Trade tracking
        self.pending_entry = None
        self.entry_price = None
        self.stop_loss = None
        self.fib_targets = []
        
    def _calculate_vortex(self, which):
        vortex = ta.vortex(
            high=self.data.High,
            low=self.data.Low,
            close=self.data.Close,
            length=self.vi_period
        )
        return vortex.iloc[:, 0] if which == 'plus' else vortex.iloc[:, 1]
    
    def next(self):
        price = self.data.Close[-1]
        print(f"🌙 Processing candle: {self.data.index[-1]} | Close: {price:.2f}")
        
        if not self.position:
            self._check_entries()
        else:
            self._manage_position()

    def _check_entries(self):
        # Check for new crossover signals
        current_vi_plus = self.vi_plus[-1]
        current_vi_minus = self.vi_minus[-1]
        prev_vi_plus = self.vi_plus[-2] if len(self.vi_plus) > 1 else current_vi_plus
        prev_vi_minus = self.vi_minus[-2] if len(self.vi_minus) > 1 else current_vi_minus
        
        volume_ok = self.data.Volume[-1] < self.volume_ma[-1]
        
        # Long condition
        if crossover(current_vi_plus, current_vi_minus) and volume_ok:
            self.pending_entry = 'long'
            self.entry_price = self.data.High[-1]
            print(f"🚀 LONG signal! VI+ crossed VI- | Entry threshold: {self.entry_price:.2f}")
            
        # Short condition
        elif crossover(current_vi_minus, current_vi_plus) and volume_ok:
            self.pending_entry = 'short'
            self.entry_price = self.data.Low[-1]
            print(f"🚀 SHORT signal! VI- crossed VI+ | Entry threshold: {self.entry_price:.2f}")
        
        # Check entry confirmation
        if self.pending_entry == 'long' and self.data.Close[-1] > self.entry_price:
            self._enter_trade(direction='long')
        elif self.pending_entry == 'short' and self.data.Close[-1] < self.entry_price:
            self._enter_trade(direction='short')

    def _enter_trade(self, direction):
        # Risk management calculations
        current_swing_high = self.swing_high[-1]
        current_swing_low = self.swing_low[-1]
        entry_price = self.data.Close[-1]
        
        if direction == '