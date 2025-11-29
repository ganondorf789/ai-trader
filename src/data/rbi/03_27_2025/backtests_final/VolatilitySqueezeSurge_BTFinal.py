I'll fix the code while maintaining the original strategy logic. Here's the debugged version with Moon Dev themed prints:

```python
# 🌙 MOON DEV BACKTESTING ENGINE - VOLATILITY SQUEEZE SURGE STRATEGY 🚀

import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# ================ DATA PREPARATION ================
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names and handle data
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
data.set_index('datetime', inplace=True)

# ================ STRATEGY IMPLEMENTATION ================
class VolatilitySqueezeSurgeStrategy(Strategy):
    risk_per_trade = 0.01  # 1% of equity per trade 🌙
    
    def init(self):
        # ========== INDICATOR SETUP ==========
        # Bollinger Bands (20,2) using TA-Lib lambda wrappers
        self.bb_upper = self.I(lambda c: talib.BBANDS(c, 20, 2, 2, 0)[0], self.data.Close)
        self.bb_middle = self.I(lambda c: talib.BBANDS(c, 20, 2, 2, 0)[1], self.data.Close)
        self.bb_lower = self.I(lambda c: talib.BBANDS(c, 20, 2, 2, 0)[2], self.data.Close)
        
        # RSI (14-period) 
        self.rsi = self.I(talib.RSI, self.data.Close, 14)
        
        # Volume 20-period SMA
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20)
        
        # ATR (14-period) for volatility measurement
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
        print("🌙✨ STRATEGY INDICATORS INITIALIZED! Ready for launch...")

    def next(self):
        # Skip initial bars without indicator data
        if len(self.data) < 20:
            return

        # ========== ENTRY CONDITIONS ==========
        current_close = self.data.Close[-1]
        
        # 1. Bollinger Band Width Contraction (<1%)
        bb_width = ((self.bb_upper[-1] - self.bb_lower[-1]) / self.bb_middle[-1]) * 100
        
        # 2. RSI Crossover (>35)
        rsi_crossover = (self.rsi[-2] < 35 and self.rsi[-1] > 35)
        
        # 3. Volume Surge (>15% of 20-period average)
        volume_condition = self.data.Volume[-1] > (self.volume_sma[-1] * 1.15)
        
        # ========== ENTRY LOGIC ==========
        if (bb_width < 1) and rsi_crossover and volume_condition and not self.position:
            # Calculate risk parameters
            atr_value = self.atr[-1]
            entry_price = current_close
            sl_price = entry_price - 0.5 * atr_value  # Stop loss at 0.5x ATR
            tp_price = entry_price + 1.5 * atr_value  # Take profit at 1.5x ATR
            
            # Position sizing calculation
            risk_amount = self.equity * self.risk_per_trade
            risk_per_share = entry_price - sl_price
            position_size = int(round(risk_amount / risk_per_share))
            
            if position_size > 0:
                self.buy(size=position_size, 
                        sl=sl_price,
                        tp=tp_price,
                        tag="🌙 VOLATILITY_SQUEEZE_ENTRY")
                
                print(f"\n🚀🌕 MOON DEV TRADE LAUNCHED! 🚀"
                      f"\nEntry Price: {entry_price:.2f}"
                      f"\nPosition Size: {position_size} units"
                      f"\nATR Value: {atr_value:.2f}"