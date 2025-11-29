```python
# 🌙 Moon Dev's FibroBandSync Backtest Implementation 🚀

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import numpy as np
import talib

class FibroBandSync(Strategy):
    def init(self):
        # 🌟 Calculate Indicators using TA-Lib
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        # Trend Identification (50-period EMA)
        self.ema50 = self.I(talib.EMA, close, 50, name='EMA 50')
        
        # Swing High/Low (20-period)
        self.swing_high = self.I(talib.MAX, high, 20, name='Swing High')
        self.swing_low = self.I(talib.MIN, low, 20, name='Swing Low')
        
        # Bollinger Bands (20-period, 2 std)
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            lambda: talib.BBANDS(close, 20, 2, 2),
            name=('BB Upper', 'BB Middle', 'BB Lower')
        )
        
        # Fibonacci Levels Calculation
        def calc_fib(swing_high, swing_low):
            diff = swing_high - swing_low
            return (
                swing_low + diff * 0.382,
                swing_low + diff * 0.5,
                swing_low + diff * 0.618
            )
        
        self.fib_levels = self.I(
            lambda: calc_fib(self.swing_high, self.swing_low),
            name=('Fib 38.2%', 'Fib 50%', 'Fib 61.8%')
        )
        
        # Bollinger Band Width
        self.bb_width = self.I(
            lambda: self.bb_upper - self.bb_lower,
            name='BB Width'
        )

    def next(self):
        # 🌙 Strategy Logic Core
        if len(self.data) < 50:  # Ensure enough data
            return

        # Current values
        price = self.data.Close[-1]
        ema50 = self.ema50[-1]
        fib_382, fib_500, fib_618 = self.fib_levels[-1]

        # 🚦 Entry Conditions
        if not self.position:
            # Trend confirmation (EMA slope up)
            ema_slope = ema50 > self.ema50[-2]
            
            # Price near Fib level (0.5% tolerance)
            near_fib = any(abs(price - level) < price*0.005 
                         for level in (fib_382, fib_500, fib_618))
            
            # Price within Bollinger Bands
            in_bb = self.bb_lower[-1] < price < self.bb_upper[-1]

            if all([ema_slope, near_fib, in_bb]):
                # 🎯 Risk Management
                risk_pct = 0.01  # 1% equity risk
                stop_loss = min(self.swing_low[-1], *self.fib_levels[-1])
                risk_distance = price - stop_loss
                
                if risk_distance > 0:
                    position_size = (self.equity * risk_pct) / risk_distance
                    position_size = int(round(position_size))
                    
                    if position_size > 0:
                        self.buy(
                            size=position_size,
                            sl=stop_loss,
                            tp=price + 2*risk_distance,  # 2:1 RR
                            exo={'fib_level': fib_500}  # Store entry context
                        )
                        print(f"🌕🚀 MOON DEV ENTRY @ {price:.2f}")
                        print(f"   Size: {position_size} | SL: {stop_loss:.2f}")

        # 🛑 Exit Conditions
        else:
            # Volatility contraction exit
            current_width = self.bb_width[-1]
            avg_width = np.mean(self.bb_width[-5:])
            
            if current_width < avg_width * 0.7:
                self.position.close()
                print(f"🌑✨ VOLATILITY DROP EXIT @ {price:.2f}")

# 🌍 Data Preparation
data = pd.read_csv(
    "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv",
    parse_dates=['datetime'],
    index_col='datetime'
)

# 🧹 Clean data columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col