Here's the complete fixed code with Moon Dev themed debug prints and proper position sizing:

```python
# 🌙 Moon Dev's VoltaicBreakout Backtest Script 🚀

import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ========================
# 🌌 DATA PREPARATION 
# ========================

# Load and clean data
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Cleanse column names 🌟
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

# Proper column mapping 🌙
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Set datetime index ✨
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# ========================
# 🌠 VOLTAIC BREAKOUT STRATEGY
# ========================

class VoltaicBreakout(Strategy):
    risk_pct = 0.01  # 🌑 1% risk per trade
    
    def init(self):
        # 🌗 Bollinger Bands (20,2)
        self.bb_upper = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)[0], self.data.Close)
        self.bb_lower = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)[2], self.data.Close)
        
        # 🌓 Volume Confirmation (5-period SMA)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=5)
        
        # 🌔 ATR for exits (14-period)
        self.atr = self.I(lambda h,l,c: talib.ATR(h,l,c,timeperiod=14), 
                         self.data.High, self.data.Low, self.data.Close)
        
        self.entry_atr = None  # 🌕 Track ATR at entry

    def next(self):
        # 🌑 Moon Dev's Core Logic 🌑
        if not self.position:
            # 🌟 Long Entry: Close > BB_Upper & Volume < SMA(5)
            if self.data.Close[-1] > self.bb_upper[-1] and self.data.Volume[-1] < self.volume_sma[-1]:
                if self.atr[-1] > 0:
                    risk_amount = self.equity * self.risk_pct
                    position_size = int(round(risk_amount / (2 * self.atr[-1])))
                    if position_size > 0:
                        self.buy(size=position_size)
                        self.entry_atr = self.atr[-1]
                        print(f"🌙✨🚀 MOON DEV LONG! Size: {position_size} @ {self.data.Close[-1]} 🚀")
            
            # 🌟 Short Entry: Close < BB_Lower & Volume < SMA(5)
            elif self.data.Close[-1] < self.bb_lower[-1] and self.data.Volume[-1] < self.volume_sma[-1]:
                if self.atr[-1] > 0:
                    risk_amount = self.equity * self.risk_pct
                    position_size = int(round(risk_amount / (2 * self.atr[-1])))
                    if position_size > 0:
                        self.sell(size=position_size)
                        self.entry_atr = self.atr[-1]
                        print(f"🌙✨🚀 MOON DEV SHORT! Size: {position_size} @ {self.data.Close[-1]} 🚀")
        else:
            # 🌗 Exit Logic: 2x ATR Trailing Stop
            if self.position.is_long:
                exit_price = self.position.entry_price - 2*self.entry_atr
                if self.data.Low[-1] <= exit_price:
                    self.position.close()
                    print(f"🌙💥 LONG EXIT! Price fell 2xATR ({2*self.entry_atr:.1f}) 💥")
            
            elif self.position.is_short:
                exit_price = self.position.entry_price + 2*self.entry_atr
                if self.data.High[-1] >= exit_price:
                    self.position.close()
                    print(f"🌙💥 SHORT EXIT! Price rose 2