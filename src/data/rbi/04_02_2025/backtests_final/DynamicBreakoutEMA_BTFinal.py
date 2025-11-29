Here's the fully debugged and Moon Dev optimized backtest code with all technical issues fixed while preserving the original strategy logic:

```python
# 🌙 Moon Dev Backtest Engine - Dynamic Breakout EMA Strategy 🌙
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Clean and prepare data moon-style 🌕
data_path = "BTC-USD-15m.csv"  # Updated path for portability
data = pd.read_csv(data_path)

# Lunar Data Cleansing Ritual 🌘
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col], errors='ignore')
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
data = data.dropna()  # Remove any cosmic debris (NaN values)

class DynamicBreakoutEMA(Strategy):
    # Cosmic Configuration 🌌
    ema50_period = 50
    ema200_period = 200
    adx_period = 14
    atr_period = 14
    swing_high_window = 20
    risk_pct = 0.01  # 1% risk per trade

    def init(self):
        # Celestial Indicators Calculation 🌠
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema50_period)
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema200_period)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_high_window)

    def next(self):
        # Lunar Trading Logic Cycle 🌑➔🌕
        price = self.data.Close[-1]
        
        if not self.position:
            # 🌟 Golden Cross Detection (Moon Dev approved implementation)
            if crossover(self.ema50, self.ema200):
                print(f"🌙 Moon Dev Alert! EMA Golden Cross Detected at {price:.2f}")
                print("✨ Cosmic Alignment: 50-EMA now above 200-EMA ✨")
            
            # Entry Conditions Check
            if (self.ema50[-1] > self.ema200[-1] and
                self.adx[-1] > 25 and
                self.adx[-1] > self.adx[-2] and
                len(self.swing_high) >= 2):
                
                # Swing High Breakout Check
                if price > self.swing_high[-2]:
                    # Cosmic Risk Management Calculations 🌍
                    atr_value = self.atr[-1]
                    entry_price = price
                    sl_price = max(0.01, entry_price - 2 * atr_value)  # Ensure positive stop
                    tp_price = entry_price + 3 * atr_value
                    
                    # Position Size Alchemy ✨ (ensuring whole units)
                    risk_amount = self.equity * self.risk_pct
                    position_size = int(round(risk_amount / (entry_price - sl_price)))
                    
                    if position_size > 0:
                        self.buy(size=position_size, sl=sl_price, tp=tp_price)
                        print(f"🚀 Moon Dev BUY Signal! 🌕 Size: {position_size} | Entry: {entry_price:.2f}")
                        print(f"   🌑 Stop Loss: {sl_price:.2f} | 🌕 Take Profit: {tp_price:.2f}")
                        print("🌌 Trade activated with celestial precision! 🌌")
        else:
            # Exit Condition: ADX Momentum Collapse
            if self.adx[-1] < 25:
                self.position.close()
                print(f"🌙 Moon Dev EXIT! Momentum Lost at {price:.2f}")
                print("🌘 Cosmic forces weakening - closing position 🌘")

# Launch Backtest Rocket 🚀
print("🌕🌖🌗🌘🌑🌒🌓🌔🌕 Moon Dev Backtest Initiation Sequence 🌕🌖🌗🌘🌑🌒🌓🌔🌕")
bt = Backtest(data, DynamicBreakout