```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION FOR LIQUIDATION REVERSAL STRATEGY 🚀

import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# =====================
# DATA PREPROCESSING 🌐
# =====================

# Load and clean data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Convert to 4H timeframe 🌓
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
data = data.resample('4H').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

# Calculate Heikin-Ashi candles 🕯️
data['ha_close'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
ha_open = [(data['open'][0] + data['close'][0]) / 2]
for i in range(1, len(data)):
    ha_open.append((ha_open[i-1] + data['ha_close'][i-1]) / 2)
data['ha_open'] = ha_open
data['ha_high'] = data[['high', 'ha_open', 'ha_close']].max(axis=1)
data['ha_low'] = data[['low', 'ha_open', 'ha_close']].min(axis=1)

# Format columns for backtesting.py 📊
data.columns = [col.capitalize() for col in data.columns]
data = data.rename(columns={
    'Ha_open': 'HA_Open',
    'Ha_close': 'HA_Close',
    'Ha_high': 'HA_High',
    'Ha_low': 'HA_Low'
})

# ======================
# STRATEGY CLASS 🧠
# ======================

class LiquidationReversal(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌱
    rr_ratio = 2     # Risk:reward ratio ⚖️
    swing_window = 50  # Swing high/low lookback
    entry_buffer = 0.005  # 0.5% buffer for zone entry
    
    def init(self):
        # Swing point detection 🔍
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_window)
        
    def next(self):
        if self.position:
            return  # Moon Dev agents never overtrade! 🌙
            
        price = self.data.Close[-1]
        equity = self.equity
        
        # Liquidation zone detection 🚨
        near_swing_high = price > (self.swing_high[-1] * (1 - self.entry_buffer))
        near_swing_low = price < (self.swing_low[-1] * (1 + self.entry_buffer))
        
        # Heikin-Ashi exhaustion check 🕯️
        current_body = abs(self.data.HA_Close[-1] - self.data.HA_Open[-1])
        prev_body = abs(self.data.HA_Close[-2] - self.data.HA_Open[-2])
        body_shrink = current_body < prev_body * 0.5
        
        # Trend reversal confirmation 🔃
        bearish_reversal = (self.data.HA_Close[-1] < self.data.HA_Open[-1] and 
                          self.data.HA_Close[-2] > self.data.HA_Open[-2])
        bullish_reversal = (self.data.HA_Close[-1] > self.data.HA_Open[-1] and 
                          self.data.HA_Close[-2] < self.data.HA_Open[-2])
        
        # Entry logic 🎯
        if near_swing_high and body_shrink and bearish_reversal:
            sl = self.swing_high[-1] * 1.005  # Stop above swing high
            risk_amount = equity * self.risk_pct
            position_size = int(round(risk_amount / (sl - price)))
            
            if position_size > 0:
                self.sell(size=position_size