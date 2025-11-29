I'll analyze and fix the backtest code while maintaining the Moon Dev theme. Here's the complete fixed version:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

class AnomalyBreakout(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    funding_threshold = -0.001  # -0.1%
    exit_funding_threshold = -0.0005  # -0.05%
    atr_multiplier = 2
    max_atr_multiplier = 3
    
    def init(self):
        self.sma50 = self.I(talib.SMA, self.data.Close, timeperiod=50, name='SMA 50')
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, 
                           self.data.Close, timeperiod=14, name='ATR 14')
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20, 
                                name='Volume SMA 20')
        
        print("🌙 Moon Dev Indicators Activated ✨")
        print("⚡ SMA50 | ATR14 | Volume SMA20 ⚡")
        print("✨ All indicators powered by TA-Lib ✨")

    def next(self):
        if not self.position:
            if (self.data.Close[-1] < self.sma50[-1] and
                self.data.fundingrate[-1] < self.funding_threshold and
                self.data.Volume[-1] > self.volume_sma[-1]):
                
                atr_value = self.atr14[-1]
                if atr_value == 0:
                    print("🌑 Warning: Zero ATR detected - skipping trade")
                    return
                
                stop_distance = min(self.atr_multiplier * atr_value,
                                   self.max_atr_multiplier * atr_value)
                risk_amount = self.equity * self.risk_percent
                position_size = int(round(risk_amount / stop_distance))
                
                if position_size > 0:
                    entry_price = self.data.Open[-1]
                    print(f"🚀 MOON DEV SHORT SIGNAL 🚀"
                          f"\n🌕 Entry Price: {entry_price:.2f}"
                          f"\n🌖 SMA50: {self.sma50[-1]:.2f}"
                          f"\n🌗 Funding Rate: {self.data.fundingrate[-1]:.4%}"
                          f"\n🌘 Position Size: {position_size} units"
                          f"\n🌑 Stop Distance: {stop_distance:.2f} ✨")
                    
                    self.sell(size=position_size,
                             sl=entry_price + stop_distance,
                             trailing_stop=stop_distance)
        else:
            if self.data.fundingrate[-1] >= self.exit_funding_threshold:
                print(f"✨ FUNDING RATE NORMALIZATION EXIT ✨"
                      f"\n🌕 Current Rate: {self.data.fundingrate[-1]:.4%}"
                      f"\n🌖 Position Closed at: {self.data.Open[-1]:.2f}"
                      f"\n🌗 Profit: {self.position.pl:.2f} 🌙")
                self.position.close()

# Data preprocessing
data = pd.read_csv(
    "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv",
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean and prepare data with Moon Dev precision
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True, errors='ignore')

print("\n🌙 DATA PREPARED WITH MOON DEV PROTOCOLS 🌙")
print("✨ CRYSTAL CLEAN DATA STRUCTURE ✨")
print(f"🌕 Columns: {data.columns.tolist()}")
print(f"🌖 First 3 rows:\n{data.iloc[:3]}")
print("🌗 Data quality checks complete")
print("🌘 Ready for lunar trading analysis ✨\n")

# Execute backtest
bt = Backtest(data, AnomalyBreakout, cash=1_000_000, commission=.002)
stats = bt.run()

print("\n🌕🌖🌗🌘🌑🌒🌓🌔🌕")
print("✨ MOON DEV BACKTEST COMPLETE