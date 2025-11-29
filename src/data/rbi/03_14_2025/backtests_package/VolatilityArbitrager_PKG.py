Here's the fixed code with all `backtesting.lib` imports and functions removed, and proper Moon Dev themed debug prints added:

```python
# 🌙 Moon Dev Backtest AI Implementation for VolatilityArbitrager Strategy 🚀

import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# =====================
# DATA PREPARATION 🌌
# =====================
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names 🧹
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

# Format columns to backtesting.py standards 📊
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert and set datetime index ⏰
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# =====================
# STRATEGY CLASS 🌗
# =====================
class VolatilityArbitrager(Strategy):
    # Strategy parameters 📈
    risk_per_trade = 0.01  # 1% risk per trade
    atr_period = 14
    hv_period = 20
    iv_period = 10
    volume_spike_multiplier = 2.0
    
    def init(self):
        # =====================
        # INDICATORS CALCULATION 🌠
        # =====================
        # Volatility Measures
        self.hv = self.I(talib.STDDEV, self.data.Close, timeperiod=self.hv_period, name='HV')
        self.iv = self.I(talib.STDDEV, self.data.Close, timeperiod=self.iv_period, name='IV')
        
        # Volume Metrics
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='Volume MA')
        
        # VWAP Calculation (custom implementation) 🌊
        def calculate_vwap(high, low, close, volume):
            typical_price = (high + low + close) / 3
            cumulative_tp = (typical_price * volume).cumsum()
            cumulative_vol = volume.cumsum()
            return cumulative_tp / cumulative_vol
            
        self.vwap = self.I(calculate_vwap, self.data.High, self.data.Low, self.data.Close, self.data.Volume, name='VWAP')
        
        # Volatility ATR for risk management 🛡️
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period, name='ATR')

    def next(self):
        # =====================
        # ENTRY LOGIC 🚪
        # =====================
        if not self.position:
            # Calculate current values
            current_iv = self.iv[-1]
            current_hv = self.hv[-1]
            spread = current_iv - current_hv
            volume_spike = self.data.Volume[-1] > (self.volume_ma[-1] * self.volume_spike_multiplier)
            price_above_vwap = self.data.Close[-1] > self.vwap[-1]
            price_below_vwap = self.data.Close[-1] < self.vwap[-1]
            atr_value = self.atr[-1]

            # Long Entry: IV < HV with confirmation 🌕
            if spread < 0 and volume_spike and price_above_vwap:
                entry_price = self.data.Close[-1]
                stop_loss = entry_price - (atr_value * 1.5)
                take_profit = entry_price + (atr_value * 3)
                
                # Position sizing calculation
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = entry_price - stop_loss
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                    print(f"🌕 MOON DEV LONG SIGNAL! Buying {position_size} units at {entry_price:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f} 🚀")

            # Short Entry: IV > HV with confirmation 🌑