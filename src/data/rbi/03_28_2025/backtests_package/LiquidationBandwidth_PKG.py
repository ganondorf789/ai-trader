Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed implementation:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION 🚀
# PACKAGE AI VERIFIED - NO BACKTESTING.LIB USAGE DETECTED ✨

# 1. IMPORT NECESSARY LIBRARIES
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import numpy as np

# 2. DATA PREPARATION
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

# 3. STRATEGY IMPLEMENTATION
class LiquidationBandwidth(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # 30-day Liquidation Cluster Level (Swing Low)
        self.liquidation_support = self.I(talib.MIN, self.data.Low, 30, name='LIQ_SUPPORT')
        
        # Bollinger Bands with 2 std dev
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name='BBANDS'
        )
        self.bb_middle = self.I(talib.SMA, self.data.Close, timeperiod=20, name='BB_MIDDLE')
        
        # Keltner Channel components
        self.kc_ema = self.I(talib.EMA, self.data.Close, timeperiod=20, name='KC_EMA')
        self.kc_atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=20, name='KC_ATR')
        
        # Swing Low for stop loss
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=5, name='SWING_LOW')
        
        # Track Bollinger Bandwidth
        self.bb_widths = []

    def next(self):
        i = len(self.data) - 1
        price = self.data.Close[i]
        
        # 1. BOLLINGER BANDWIDTH CALCULATION 🌙
        if i >= 20:
            current_width = (self.bb_upper[i] - self.bb_lower[i]) / self.bb_middle[i]
            self.bb_widths.append(current_width)
        else:
            self.bb_widths.append(0)
        
        # 2. ENTRY LOGIC ✨
        if not self.position:
            # Liquidation cluster proximity check (1% threshold)
            liq_support = self.liquidation_support[i]
            support_zone = liq_support * 0.99 <= price <= liq_support * 1.01
            
            # Bollinger bandwidth contraction check
            bandwidth_low = False
            if len(self.bb_widths) >= 10:
                bandwidth_window = self.bb_widths[-10:]
                bandwidth_low = self.bb_widths[-1] == min(bandwidth_window)
            
            if support_zone and bandwidth_low:
                # 3. RISK MANAGEMENT CALCULATIONS 🌙
                stop_price = min(self.swing_low[i], liq_support * 0.98)
                risk_amount = self.equity * self.risk_per_trade
                risk_per_unit = price - stop_price
                
                if risk_per_unit > 0:
                    position_size = int(round(risk_amount / risk_per_unit))
                    self.buy(size=position_size, sl=stop_price)
                    print(f"🌙✨ MOON SIGNAL: LONG @ {price:.2f}")
                    print(f"🚀 Size: {position_size} | 💥 SL: {stop_price:.2f}")

        # 4. EXIT LOGIC 🚀
        else:
            # Keltner Channel upper band
            kc_upper = self.kc_ema[i] + 1.5