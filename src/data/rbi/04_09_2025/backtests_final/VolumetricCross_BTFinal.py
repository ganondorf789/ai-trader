Here's the debugged backtest code with Moon Dev themed enhancements while preserving the original strategy logic:

```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR VOLUMETRIC CROSS STRATEGY 🚀

from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# ========================
# 🌌 DATA PREPARATION 
# ========================
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Cleanse column names with Moon Dev precision ✨
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

# Format columns for backtesting.py requirements 🌙
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Convert and set datetime index 🕰️
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# ========================
# 🌠 VOLUMETRIC CROSS STRATEGY
# ========================
class VolumetricCross(Strategy):
    ema_short = 50
    ema_long = 200
    rsi_period = 14
    vol_ma_period = 20  # Fixed typo in variable name
    risk_pct = 0.01  # 1% risk per trade 🌕
    
    def init(self):
        # 🌗 Calculate core indicators using TA-Lib
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_short)
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_long)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_ma_period)
        
        # Track crossover state like Moon phases 🌑🌒🌓
        self.last_ema_state = False
        
    def next(self):
        # Wait until indicators are fully loaded ⏳
        if len(self.data) < max(self.ema_long, self.vol_ma_period) + 1:
            return
        
        # ========================
        # 🚀 ENTRY LOGIC 
        # ========================
        current_close = self.data.Close[-1]
        current_vol = self.data.Volume[-1]
        
        # EMA crossover detection 🌗→🌕 (manual implementation)
        ema_crossover = (self.ema50[-2] < self.ema200[-2]) and \
                        (self.ema50[-1] > self.ema200[-1])
            
        # Volume spike calculation 🌋
        vol_spike = current_vol > (self.vol_ma[-1] * 1.5)
        
        if not self.position and ema_crossover and vol_spike:
            # 🌙 Calculate position size with Moon math 🌌
            sl_pct = 0.02  # 2% stop loss
            risk_amount = self.risk_pct * self.equity
            risk_per_share = current_close * sl_pct
            position_size = int(round(risk_amount / risk_per_share))
            
            if position_size > 0:
                self.buy(
                    size=position_size,
                    sl=current_close * (1 - sl_pct),
                    tag="Moon_Entry 🌕"
                )
                print(f"🌙✨ MOON DEV ALERT: Long entry at {current_close:.2f} | Size: {position_size} 🚀")
                
        # ========================
        # 📉 EXIT LOGIC 
        # ========================
        if self.position:
            # RSI exit condition 📉
            if self.rsi[-1] < 70 and self.rsi[-2] >= 70:
                self.position.close()
                print(f"🌙💫 MOON DEV WARNING: RSI exit at {current_close:.2f} | Preserving gains! 📉")

# ========================
# 🚀 BACKTEST EXECUTION
# ========================
bt = Backtest(data, VolumetricCross, cash=1_000_000, exclusive_orders=True)

print("🌙✨ MOON DEV BACKTEST INITIATED 🚀\n")
stats = bt.run()
print("\n🌙💎 BACK