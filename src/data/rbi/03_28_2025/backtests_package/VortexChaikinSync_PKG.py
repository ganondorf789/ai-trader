Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed debug prints added:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Data preparation
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
})

class VortexChaikinSync(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    swing_period = 20
    
    def init(self):
        # Vortex Indicator
        self.vi_plus, self.vi_minus = self.I(
            lambda: talib.VORTEX(self.data.High, self.data.Low, self.data.Close, 14),
            name=['VI+', 'VI-']
        )
        
        # Chaikin Oscillator
        self.chaikin = self.I(talib.ADOSC, self.data.High, self.data.Low, self.data.Close, 
                              self.data.Volume, 3, 10, name='Chaikin')
        
        # Swing Low for trailing stops
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period, name='Swing Low')
        
        print("🌙✨ Moon Dev System Initialized")
        print(f"Vortex(14) | Chaikin(3,10) | Swing({self.swing_period}) | Risk: {self.risk_pct*100}%")

    def next(self):
        price = self.data.Close[-1]
        
        # Entry Logic 🌙🚀
        if not self.position:
            # Vortex crossover condition (replaced backtesting.lib.crossover)
            vi_bullish = (self.vi_plus[-2] < self.vi_minus[-2]) and (self.vi_plus[-1] > self.vi_minus[-1])
            
            # Chaikin confirmation
            chaikin_rising = self.chaikin[-1] > self.chaikin[-2]
            chaikin_positive = self.chaikin[-1] > 0
            
            if vi_bullish and chaikin_rising and chaikin_positive:
                # Risk management calculations
                entry_price = self.data.Open[-1]  # Next candle's open
                risk_amount = self.equity * self.risk_pct
                initial_stop = entry_price * (1 - self.risk_pct)
                trailing_stop = self.swing_low[-1]
                effective_stop = max(initial_stop, trailing_stop)
                stop_distance = entry_price - effective_stop
                
                if stop_distance <= 0:
                    print(f"⚠️ Moon Dev Alert: Invalid stop at {effective_stop:.2f} (Entry: {entry_price:.2f})")
                    return
                
                position_size = int(round(risk_amount / stop_distance))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=effective_stop, 
                            tag=f"🌙 ENTRY | Size: {position_size} | Stop: {effective_stop:.2f}")
                    print(f"🚀🌙 Moon Dev LONG @ {entry_price:.2f} | Size: {position_size}")

        # Exit Logic 🌙🔻
        if self.position:
            # Multiple exit conditions
            exit_price = self.data.Open[-1]
            
            # 1. Vortex crossover down (replaced backtesting.lib.crossover)
            vi_bearish = (self.vi_minus[-2] < self.vi_plus[-2]) and (self.vi_minus[-1] > self.vi_plus[-1])
            
            # 2. Chaikin below zero
            chaikin_negative = self.chaikin[-1] < 0
            
            # 3. Bearish divergence check
            price_higher_high = self.data.High[-1] > self.data.High[-2]
            chaikin_lower_high = self.chaikin[-1] < self.chaikin[-2]
            divergence = price_higher_high and chaikin_lower_high
            
            # 4. Time exit (5 bars)
            time_exit = len(self.data) -