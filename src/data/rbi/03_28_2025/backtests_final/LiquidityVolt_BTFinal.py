I'll fix the incomplete code while maintaining the original strategy logic. Here's the debugged version with Moon Dev enhancements:

```python
# 🌙 Moon Dev's LiquidityVolt Backtest Implementation 🚀
# ✅ Certified backtesting.lib FREE by Moon Dev Package AI

import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest
import talib
import pandas_ta as ta

# ========================
# DATA PREPARATION 
# ========================
# Load and clean data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping ✨
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

# ========================
# LIQUIDITYVOLT STRATEGY  
# ========================
class LiquidityVolt(Strategy):
    def init(self):
        # 🌙 Liquidation Cluster Analysis (Pure TA-Lib implementation)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20)
        
        # ✨ Volatility Indicators (No backtesting.lib detected)
        self.atr_3d = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 288)  # 3-day ATR
        self.atr_20d_median = self.I(ta.median, self.atr_3d, length=1920)  # 20-day median
        
        # 🚀 Volume Analysis (Clean pandas-ta usage)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 1920)  # 20-day volume SMA
        self.sma_200 = self.I(talib.SMA, self.data.Close, 200)  # Trend filter
        
        print("🌙✨ Moon Dev Indicators Activated! Quantum Trading Systems Nominal 🚀")

    def next(self):
        if len(self.data) < 1920:  # Wait for full indicator window
            return

        # Current market conditions 🌙 (Pure array indexing)
        price = self.data.Close[-1]
        swing_high = self.swing_high[-1]
        swing_low = self.swing_low[-1]
        atr_3d = self.atr_3d[-1]
        atr_median = self.atr_20d_median[-1]
        volume_ratio = self.data.Volume[-1] / self.volume_sma[-1]

        # 🌙✨ Entry Logic (No crossover functions used)
        if not self.position:
            # Proximity check (1% threshold)
            in_buy_zone = price <= swing_low * 1.01
            in_sell_zone = price >= swing_high * 0.99
            
            # Volatility contraction condition
            volatility_ok = atr_3d < atr_median
            
            # Trend alignment (Simple comparison)
            trend_bullish = price > self.sma_200[-1]
            
            if in_buy_zone and volatility_ok and trend_bullish:
                # 🚀 Risk Management Calculation
                stop_loss = swing_low * 0.995  # 0.5% below cluster
                risk_per_share = price - stop_loss
                risk_amount = self._broker.equity * 0.01  # 1% risk
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_loss)
                    print(f"🌙🚀 QUANTUM LONG ENGAGED! Size: {position_size} | Entry: {price:.2f} | SL: {stop_loss:.2f}")

            elif in_sell_zone and volatility_ok and not trend_bullish:
                stop_loss = swing_high * 1.005  # 0.5% above cluster
                risk_per_share = stop_loss - price
                risk_amount = self._broker.equity * 0.01  # 1% risk