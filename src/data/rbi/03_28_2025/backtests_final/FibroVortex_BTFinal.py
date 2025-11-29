I'll fix the incomplete code while maintaining the original strategy logic. Here's the debugged version with Moon Dev themed prints:

```python
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import pandas_ta as ta

# DATA PREPARATION
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
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

class FibroVortex(Strategy):
    def init(self):
        # 1. Swing Calculations 🌙
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20)
        
        # 2. Vortex Indicator ✨
        self.vi_plus, self.vi_minus = self.I(ta.vi, self.data.High, self.data.Low, self.data.Close, length=14)
        
        # 3. ATR for Dynamic Stops 🛡️
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        print("🌙✨ Initialized FibroVortex with Swing(20), VI(14), ATR(14)")

    def next(self):
        current_close = self.data.Close[-1]
        swh = self.swing_high[-1]
        swl = self.swing_low[-1]
        
        if swh == swl:  # Avoid invalid levels
            print("🌙⚠️ Warning: Swing high equals swing low - skipping iteration")
            return

        # FIB CALCULATIONS 🌉
        fib_range = swh - swl
        fib_levels = {
            '382': swl + 0.382*fib_range,
            '500': swl + 0.5*fib_range,
            '618': swl + 0.618*fib_range,
            '786': swl + 0.786*fib_range,
            '236': swl + 0.236*fib_range
        }

        # PRICE PROXIMITY CHECK 🔍
        near_fib = any(abs(current_close - fib_levels[key])/fib_levels[key] <= 0.01 
                   for key in ['382', '500', '618'])

        # VOLTEX SIGNALS 🌪️ (Manual crossover detection)
        vi_cross_up = (self.vi_plus[-2] < self.vi_minus[-2]) and (self.vi_plus[-1] > self.vi_minus[-1])
        vi_cross_down = (self.vi_minus[-2] < self.vi_plus[-2]) and (self.vi_minus[-1] > self.vi_plus[-1])
        vol_diff = (self.vi_plus[-1] - self.vi_minus[-1]) < (self.vi_plus[-2] - self.vi_minus[-2])

        # RISK MANAGEMENT 💼
        risk_amount = self.equity * 0.01  # 1% risk
        
        if not self.position:
            # LONG ENTRY LOGIC 🚀
            if near_fib and vi_cross_up and vol_diff:
                sl = fib_levels['786'] - 0.5*self.atr[-1]
                risk_per_unit = current_close - sl
                if risk_per_unit > 0:
                    size = int(round(risk_amount / risk_per_unit))
                    if size > 0:
                        self.buy(size=size, sl=sl, tp=swh)
                        print(f"🚀🌙 LONG | Price: {current_close:.2f} | Size: {size} | SL: {sl:.2f} | TP: {swh:.2f}")

            # SHORT ENTRY LOGIC 🌑
            elif near_fib and vi_cross_down and vol_diff:
                sl = fib_levels['236'] + 0.5*self.atr[-1]
                risk_per_unit = sl - current_close
                if risk_per_unit > 0:
                    size = int(round(