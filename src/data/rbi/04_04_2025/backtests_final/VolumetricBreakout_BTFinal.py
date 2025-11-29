I'll fix the backtest code while maintaining the original strategy logic. Here's the corrected version with Moon Dev themed debug prints:

```python
# 🌙 Moon Dev's Volumetric Breakout Backtest 🌙
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Clean and prepare data
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean columns according to Moon Dev specs
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolumetricBreakout(Strategy):
    risk_per_trade = 0.01  # 1% equity risk
    
    def init(self):
        # 🌙 Moon Dev Indicators 🌙
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=50, name='EMA50')
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=200, name='EMA200')
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='Volume SMA')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR14')
        
        self.entry_price = None
        self.initial_sl = None
        self.trailing_high = None

    def next(self):
        # Wait until enough data
        if len(self.data) < 200:
            return
            
        # 🚀 Moon Dev Entry Logic �
        if not self.position:
            # Replaced crossover with manual check
            golden_cross = (self.ema50[-2] < self.ema200[-2]) and (self.ema50[-1] > self.ema200[-1])
            volume_spike = self.data.Volume[-1] > self.vol_sma[-1]
            
            if golden_cross and volume_spike:
                atr_value = self.atr[-1]
                if atr_value == 0:
                    print("🌙 Warning: Zero ATR value detected!")
                    return
                
                # Calculate Moon Dev Risk Management 🌙
                risk_amount = self.equity * self.risk_per_trade
                stop_distance = 2 * atr_value
                position_size = int(round(risk_amount / stop_distance))
                
                if position_size > 0:
                    self.entry_price = self.data.Close[-1]
                    self.initial_sl = self.entry_price - stop_distance
                    self.trailing_high = self.entry_price
                    
                    # 🌙 Moon Dev Themed Entry Print 🌙
                    print(f"\n🚀 MOON DEV ALERT: LONG SIGNAL DETECTED 🚀")
                    print(f"⏰ {self.data.index[-1]}")
                    print(f"💵 Entry Price: {self.entry_price:.2f}")
                    print(f"📈 Position Size: {position_size} units")
                    print(f"🛑 Initial SL: {self.initial_sl:.2f}")
                    print(f"📉 ATR Stop Distance: {stop_distance:.2f}")
                    
                    self.buy(size=position_size)

        # ✨ Moon Dev Exit Logic ✨
        elif self.position.is_long:
            current_high = self.data.High[-1]
            self.trailing_high = max(self.trailing_high, current_high)
            
            # Calculate dynamic stops
            trailing_sl = self.trailing_high * 0.95
            current_sl = max(self.initial_sl, trailing_sl)
            
            if self.data.Low[-1] <= current_sl:
                self.position.close()
                print(f"\n🌙 MOON DEV EXIT SIGNAL 🌙")
                print(f"⏰ {self.data.index[-1]}")
                print(f"💰 Exit Price: {self.data.Close[-1]:.2f}")
                print(f"📉 Final SL: {current_sl:.2f}")
                print(f"✨ P/L: {self.position.pl:.2f}")
                print("="*50)

# 🌙 Run Moon Dev Backtest 🌙
bt = Backtest(data, VolumetricBreak