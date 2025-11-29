```python
# 🌙 MOON DEV BACKTESTING SCRIPT - LIQUIDATION REVERSAL STRATEGY 🌙

import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# =====================
# DATA PREPARATION
# =====================
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and format columns 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# =====================
# STRATEGY IMPLEMENTATION
# =====================
class LiquidationReversal(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌙
    swing_period = 20
    atr_period = 14
    rejection_threshold = 0.7  # 70% wick dominance
    
    def init(self):
        # 🌙 CORE INDICATORS
        self.swing_highs = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period, name='SWING HIGH')
        self.swing_lows = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period, name='SWING LOW')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR')
        self.atr_sma = self.I(talib.SMA, self.atr, timeperiod=20, name='ATR SMA')
        
    def next(self):
        # 🌙 MOON DEV DEBUG CHECKS
        if len(self.data) < 50:  # Warmup period
            return
            
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        current_close = self.data.Close[-1]
        
        # 🌙 VOLATILITY FILTER
        atr_val = self.atr[-1]
        atr_sma_val = self.atr_sma[-1]
        volatility_compressed = atr_val < atr_sma_val
        
        # 🌙 LIQUIDATION ZONE DETECTION
        upper_zone = self.swing_highs[-1]
        lower_zone = self.swing_lows[-1]
        
        # 🌙 REJECTION PATTERN DETECTION
        candle_range = current_high - current_low
        bear_rejection = (current_high - current_close)/candle_range >= self.rejection_threshold
        bull_rejection = (current_close - current_low)/candle_range >= self.rejection_threshold
        
        # 🌙🌙🌙 ENTRY LOGIC 🌙🌙🌙
        if not self.position and volatility_compressed:
            # SHORT ENTRY: Upper zone rejection
            if current_high >= 0.99 * upper_zone and bear_rejection:
                sl_price = upper_zone * 1.005  # Allow stop-hunting buffer
                risk_distance = sl_price - self.data.Close[-1]
                position_size = self._calculate_position_size(risk_distance)
                
                if position_size > 0:
                    self.sell(size=position_size, 
                             sl=sl_price,
                             tp=self.data.Close[-1] - 2*risk_distance)
                    print(f"🌙✨ BEARISH REVERSAL! Short {position_size} units | "
                          f"Entry: {self.data.Close[-1]:.2f}, SL: {sl_price:.2f} "
                          f"Risk: {risk_distance:.2f} ({self.risk_percent*100}% equity)")
            
            # LONG ENTRY: Lower zone rejection
            elif current_low <= 1.01 * lower_zone and bull_rejection:
                sl_price = lower_zone * 0.995  # Allow stop-hunting buffer
                risk_distance = self.data.Close[-1] - sl_price
                position_size = self._calculate_position_size(risk_distance)
                
                if position_size > 0:
                    self.buy(size=position_size, 
                            sl=sl_price,
                            tp=self.data.Close[-1] + 2*risk