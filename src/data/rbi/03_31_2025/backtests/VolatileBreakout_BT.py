```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR VOLATILEBREAKOUT STRATEGY 🌙

# ===== REQUIRED IMPORTS =====
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# ===== DATA PREPARATION =====
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
                   parse_dates=['datetime'], index_col='datetime')

# 🌙 MOON DEV DATA CLEANSING RITUAL 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# ===== STRATEGY IMPLEMENTATION =====
class VolatileBreakout(Strategy):
    risk_pct = 0.01  # 🌙 1% risk per trade as per Moon Dev standards 🌙
    
    def init(self):
        # 🌌 COSMIC INDICATOR CALCULATIONS 🌌
        self.atr20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 20)
        self.atr90_avg = self.I(talib.SMA, self.atr20, 90)
        self.volume_ma30 = self.I(talib.SMA, self.data.Volume, 30)
        self.recent_high = self.I(talib.MAX, self.data.High, 20)
        self.recent_low = self.I(talib.MIN, self.data.Low, 20)
        
        print("✨ MOON DEV INDICATORS INITIALIZED ✨")

    def next(self):
        # 🌑 MOON PHASE CHECK - ONLY TRADE WHEN INDICATORS ARE READY 🌑
        if len(self.atr20) < 90 or len(self.recent_high) < 20:
            return

        # 🌕 VOLATILITY CONFIRMATION 🌕
        atr_condition = self.atr20[-1] > self.atr90_avg[-1]
        # 💧 VOLUME SURGE DETECTION 💧
        volume_condition = self.data.Volume[-1] > self.volume_ma30[-1]

        # 🚀 LONG BREAKOUT SIGNAL DETECTION 🚀
        if not self.position and atr_condition and volume_condition:
            long_trigger = self.data.High[-1] > self.recent_high[-2]
            short_trigger = self.data.Low[-1] < self.recent_low[-2]

            if long_trigger:
                self.enter_long()
            elif short_trigger:
                self.enter_short()

    def enter_long(self):
        # 🌙 LUNAR POSITION SIZING CALCULATIONS 🌙
        entry_price = self.data.Close[-1]  # Closing price breakout confirmation
        stop_loss = self.recent_high[-2]  # Previous swing high
        risk_per_share = entry_price - stop_loss
        
        if risk_per_share <= 0:
            print("🌑 MOON DEV ABORT: Negative risk detected!")
            return

        # 🪐 ASTROLOGICAL RISK MANAGEMENT 🪐
        risk_amount = self.equity * self.risk_pct
        position_size = risk_amount / risk_per_share
        position_size = int(round(position_size))
        
        if position_size <= 0:
            print("🌒 CRESCENT WARNING: Position size too small!")
            return

        take_profit = entry_price + 2 * risk_per_share
        
        # 🚀 EXECUTE LUNAR LAUNCH SEQUENCE 🚀
        self.buy(size=position_size, 
                sl=stop_loss,
                tp=take_profit,
                tag="MoonDev Long Position")
        
        print(f"🌕 MOON DEV LONG ENTRY 🌕 | Size: {position_size} | Entry: {entry_price:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")

    def enter_short(self):
        # 🌑 DARK SIDE POSITION SIZING 🌑
        entry_price = self.data.Close[-1]  # Closing price breakdown confirmation
        stop_loss = self.recent_low[-2]  # Previous swing low
        risk_per_share = stop