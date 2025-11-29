```python
# 🌙✨ Moon Dev's BandStoch Fusion Backtest Script 🚀

import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Clean and prepare moon data 🌑➡️🌕
def prepare_data(path):
    data = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    return data

class BandStochFusion(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌱
    bb_period = 20
    stoch_period = 14
    swing_window = 20
    
    def init(self):
        # 🌗 Bollinger Bands Calculation
        def bb_upper(close):
            upper, _, _ = talib.BBANDS(close, timeperiod=self.bb_period, nbdevup=2)
            return upper
        self.upper = self.I(bb_upper, self.data.Close)
        
        def bb_middle(close):
            _, middle, _ = talib.BBANDS(close, timeperiod=self.bb_period)
            return middle
        self.middle = self.I(bb_middle, self.data.Close)
        
        def bb_lower(close):
            _, _, lower = talib.BBANDS(close, timeperiod=self.bb_period, nbdevdn=2)
            return lower
        self.lower = self.I(bb_lower, self.data.Close)
        
        # 🌓 Stochastic Oscillator
        def stoch_k(high, low, close):
            k, _ = talib.STOCH(high, low, close,
                              fastk_period=self.stoch_period,
                              slowk_period=3,
                              slowd_period=3)
            return k
        self.stoch_k = self.I(stoch_k, self.data.High, self.data.Low, self.data.Close)
        
        def stoch_d(high, low, close):
            _, d = talib.STOCH(high, low, close,
                              fastk_period=self.stoch_period,
                              slowk_period=3,
                              slowd_period=3)
            return d
        self.stoch_d = self.I(stoch_d, self.data.High, self.data.Low, self.data.Close)
        
        # 🌑 Swing High/Low for Stops
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_window)
        
        print("🌙✨ Lunar Indicators Activated! Ready for Cosmic Trading! 🚀")

    def next(self):
        price = self.data.Close[-1]
        
        # 🌕 Long Exit Conditions
        if self.position.is_long:
            if (self.data.High[-1] >= self.middle[-1] or
                crossunder(self.stoch_k, 80)):
                print(f"🌗 Closing LONG at {price:.2f} (Middle Band/Stoch Exit)")
                self.position.close()
        
        # 🌑 Short Exit Conditions
        elif self.position.is_short:
            if (self.data.Low[-1] <= self.middle[-1] or
                crossover(self.stoch_k, 20)):
                print(f"🌘 Closing SHORT at {price:.2f} (Middle Band/Stoch Exit)")
                self.position.close()
                
        # Entry Signals Only When Flat
        if not self.position:
            # 🌕 Long Entry Logic
            long_cond = (
                self.data.Low[-1] <= self.lower[-1] and
                self.stoch_k[-1] < 20 and
                crossover(self.stoch_k, self.stoch_d)
            )
            
            if long_cond:
                sl = min(self.swing_low[-1], self.lower[-1])
                risk_amount = self.risk_pct * self.equity
                size = int(round(risk_amount / (price - sl)))
                if size > 0:
                    print(f"🚀🌕 BLAST OFF LONG! Entry: {price:.2f}, Size: {size}")
                    self.buy(size=size, sl=sl, tp=price + 2*(price - sl))
            
            # 🌑 Short Entry Logic
            short_cond = (