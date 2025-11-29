Here's the fully debugged and optimized backtest code with Moon Dev themed improvements:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# 🌙 MOON DEV DATA PREP STARTS HERE ✨
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()], errors='ignore')

# Proper column formatting for Backtrader
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'datetime': 'Date'
}, inplace=True)

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
# 🌙 DATA PREP COMPLETE! ✨

class DynamicBreakoutMomentum(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌙 MOON DEV INDICATOR INITIALIZATION ✨
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=50, name='EMA 50')
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=200, name='EMA 200')
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, 14, name='ADX 14')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR 14')
        self.rsi5 = self.I(talib.RSI, self.data.Close, 5, name='RSI 5')
        self.recent_high = self.I(talib.MAX, self.data.High, 20, name='20-period High')
        
        self.highest_high = None  # For trailing stop calculation
        print("🌕 MOON DEV: Indicators initialized successfully!")

    def next(self):
        # Wait for sufficient data
        if len(self.data) < 200:
            return

        # Entry conditions
        if not self.position:
            # Proper crossover implementation
            golden_cross = crossover(self.ema50, self.ema200)
            
            # ADX strength condition
            adx_rising = self.adx[-1] > self.adx[-2] and self.adx[-1] > 25
            
            # Breakout condition
            price_breakout = self.data.Close[-1] > self.recent_high[-1]

            if golden_cross and adx_rising and price_breakout:
                # 🌙 MOON DEV RISK CALCULATION ✨
                atr_value = self.atr[-1]
                if atr_value <= 0:
                    return
                
                risk_amount = self.equity * self.risk_percent
                stop_distance = 2 * atr_value
                position_size = risk_amount / stop_distance
                
                # Proper position sizing - round to whole units
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    self.highest_high = self.data.High[-1]
                    print(f"🌕 MOON DEV ALERT: Long entry at {self.data.Close[-1]:.2f}!")
                    print(f"🚀 Size: {position_size} units | Risk: ${risk_amount:.2f} ({self.risk_percent*100}% equity)")

        # Exit management
        elif self.position:
            # Update trailing stop
            self.highest_high = max(self.highest_high, self.data.High[-1])
            trail_stop = self.highest_high - 2 * self.atr[-1]
            
            # Trailing stop exit
            if self.data.Low[-1] < trail_stop:
                self.position.close()
                print(f"🌙 MOON DEV EXIT: Trailing stop triggered at {trail_stop:.2f}!")
                print(f"💎 Profit: ${self.position.pl:.2f} ({self.position.pl_pct:.2%})")
            
            # RSI overbought exit
            elif self.rsi5[-1] > 70:
                self.position.close()
                print(f"🚨 MOON DEV EXIT: RSI {self.rsi5