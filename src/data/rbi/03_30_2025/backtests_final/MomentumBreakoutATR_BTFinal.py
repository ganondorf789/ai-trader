I'll fix the code while maintaining the original strategy logic. Here's the debugged version with Moon Dev themed improvements:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Data preparation and cleaning
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

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

class MomentumBreakoutATR(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌊
    
    def init(self):
        # Moon Dev Indicators Setup 🌙✨
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=50)
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=200)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.recent_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        
        # Trade state tracking variables 🚀
        self.highest_high = None
        self.trailing_stop = None

    def next(self):
        # Moon Dev Core Strategy Logic 🌙💫
        if not self.position:
            # Entry Conditions Check 📈
            golden_cross = (self.ema50[-2] < self.ema200[-2]) and (self.ema50[-1] > self.ema200[-1])
            price_above_emas = (self.data.Close[-1] > self.ema50[-1]) and (self.data.Close[-1] > self.ema200[-1])
            adx_rising = (self.adx[-1] > 25) and (self.adx[-1] > self.adx[-2])
            breakout = self.data.Close[-1] > self.recent_high[-1]
            
            if golden_cross and price_above_emas and adx_rising and breakout:
                # Moon Dev Risk Calculation 🌙🛡️
                entry_price = self.data.Close[-1]
                atr_value = self.atr[-1]
                risk_per_share = 2 * atr_value  # 2xATR stop
                
                if risk_per_share <= 0:
                    print("🌙 MOON DEV ALERT! Zero risk detected - skipping entry")
                    return
                
                position_size = int(round((self.equity * self.risk_percent) / risk_per_share))
                
                if position_size > 0:
                    # Moon Dev Entry Signal 🚀🌙
                    self.buy(size=position_size)
                    print(f"\n🚀🌙 MOON DEV LONG ENTRY! ✨")
                    print(f"📈 Size: {position_size} units @ {entry_price:.2f}")
                    print(f"🔑 EMA50: {self.ema50[-1]:.2f} | EMA200: {self.ema200[-1]:.2f}")
                    print(f"📊 ADX: {self.adx[-1]:.2f} | ATR: {atr_value:.2f}")
                    
                    # Initialize trailing stop
                    self.highest_high = self.data.High[-1]
                    self.trailing_stop = self.highest_high - 2 * atr_value
        else:
            # Moon Dev Exit Management 🌙🔒
            self.highest_high = max(self.highest_high, self.data.High[-1])
            new_stop = self.highest_high - 2 * self.atr[-1]
            self.trailing_stop = max(self.trailing_stop, new_stop)
            
            # Check exit condition
            if self.data.Low[-1] <= self.trailing_stop:
                self.position.close()
                print(f"\n🌙✨ MOON DEV TRAILING EXIT! 🌟")
                print(f"🔒 Stop: {self.trailing_stop:.2f