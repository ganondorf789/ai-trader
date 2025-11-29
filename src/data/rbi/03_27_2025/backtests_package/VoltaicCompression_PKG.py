Here's the fixed code with proper Moon Dev themed indicators and no backtesting.lib usage:

```python
# -*- coding: utf-8 -*-
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

# =====================
# DATA PREPARATION
# =====================
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})[['Open', 'High', 'Low', 'Close', 'Volume']]

# =====================
# VOLTAIC COMPRESSION STRATEGY
# =====================
class VoltaicCompression(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # =====================
        # INDICATOR SETUP ✨
        # =====================
        # Trend Strength
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        # Bollinger Bands System 🎯
        upper, middle, lower = talib.BBANDS(self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2)
        self.bb_upper = self.I(lambda: upper, name='BB_UPPER')
        self.bb_mid = self.I(lambda: middle, name='BB_MID')
        self.bb_lower = self.I(lambda: lower, name='BB_LOWER')
        self.bbw = self.I(lambda: (upper - lower)/middle, name='BBW')
        self.bbw_sma = self.I(talib.SMA, self.bbw, timeperiod=20)
        
        # Keltner System 🌙
        self.keltner_ema = self.I(talib.EMA, self.data.Close, timeperiod=20)
        self.keltner_atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=20)
        self.keltner_upper = self.I(lambda: self.keltner_ema + 1.5*self.keltner_atr, name='KELT_UPPER')
        self.keltner_lower = self.I(lambda: self.keltner_ema - 1.5*self.keltner_atr, name='KELT_LOWER')
        
        # Risk Management Tools 🔐
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20)
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        print("🌙✨ Voltaic Compression Engaged! Lunar Markets Ready for Capture 🚀")

    def next(self):
        # =====================
        # ENTRY LOGIC 🎯
        # =====================
        if not self.position:
            # Long Entry Constellation 🌌
            if (self.adx[-1] > 25 and 
                self.bbw[-1] < self.bbw_sma[-1] and 
                self.data.Close[-1] > self.bb_upper[-1]):
                
                risk_amount = self.equity * self.risk_per_trade
                entry_price = self.data.Close[-1]
                sl_atr = entry_price - 1.5 * self.atr14[-1]
                sl_price = max(self.swing_low[-1], sl_atr)
                risk_per_unit = entry_price - sl_price
                
                if risk_per_unit > 0:
                    position_size = int(round(risk_amount / risk_per_unit))
                    print(f"🌙🚀 BULLISH IMPULSE DETECTED! Entering LONG {position_size} units ✨")
                    self.buy(size=position_size, sl=sl_price)
            
            # Short Entry Constellation 🌌
            elif (self.adx[-1] > 25 and 
                  self.bbw[-1] < self