I'll fix the code by removing all `backtesting.lib` imports and replacing crossover/crossunder functions with manual checks. Here's the corrected version with Moon Dev themed debug prints:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# Data handling
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and prepare data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VortexVolumePullback(Strategy):
    risk_percent = 0.01
    swing_period = 20
    
    def init(self):
        # Calculate Vortex Indicator using pandas_ta
        vi = ta.vortex(
            high=self.data.High, 
            low=self.data.Low, 
            close=self.data.Close, 
            length=14
        )
        self.vi_plus = self.I(lambda: vi['VIPT_14'], name='VI+')
        self.vi_minus = self.I(lambda: vi['VIMT_14'], name='VI-')
        
        # On-Balance Volume
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume, name='OBV')
        
        # Trend SMA
        self.sma50 = self.I(talib.SMA, self.data.Close, 50, name='SMA50')
        
        # Swing points for SL
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period, name='SwingHigh')
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period, name='SwingLow')
        
    def next(self):
        if self.position:
            self.manage_positions()
        else:
            self.check_entries()
    
    def manage_positions(self):
        price = self.data.Close[-1]
        
        if self.position.is_long:
            # Exit on VI crossover or OBV divergence
            if (self.vi_plus[-2] > self.vi_minus[-2] and self.vi_plus[-1] < self.vi_minus[-1]):  # Bearish crossover
                self.position.close()
                print(f"🌙✨ LONG Exit: VI bearish crossover at {price:.2f}")
            elif self.obv[-1] < self.obv[-4]:
                self.position.close()
                print(f"🌙📉 LONG Exit: OBV bearish divergence at {price:.2f}")
                
        elif self.position.is_short:
            if (self.vi_minus[-2] < self.vi_plus[-2] and self.vi_minus[-1] > self.vi_plus[-1]):  # Bullish crossover
                self.position.close()
                print(f"🌙✨ SHORT Exit: VI bullish crossover at {price:.2f}")
            elif self.obv[-1] > self.obv[-4]:
                self.position.close()
                print(f"🌙📈 SHORT Exit: OBV bullish divergence at {price:.2f}")
    
    def check_entries(self):
        price = self.data.Close[-1]
        prev_close = self.data.Close[-2]
        sma50 = self.sma50[-1]
        
        # Long Entry Criteria
        if (self.vi_plus[-1] > self.vi_minus[-1] and 
            price > sma50 and 
            self.data.Low[-1] <= sma50 and 
            self.obv[-1] > self.obv[-2]):
            
            sl = self.swing_low[-1]
            risk_share = price - sl
            if risk_share <= 0: return
            
            size = self.calculate_size(risk_share)
            if size > 0:
                self.buy(size=size)
                tp = price + 2 * risk_share
                print(f"🚀🌙 LONG Entry at {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
        
        # Short Entry Criteria
        elif (self.vi_minus[-1] > self.vi_plus[-