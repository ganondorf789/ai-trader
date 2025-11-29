I'll fix the code by removing all `backtesting.lib` imports and replacing crossover/crossunder functions with manual implementations. Here's the corrected version with Moon Dev themed debug prints:

```python
import pandas as pd
import talib
import pandas_ta as pta
from backtesting import Backtest, Strategy

# Load and preprocess data
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean column names and format
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class VortexChopFlip(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Calculate indicators using TA-Lib/pandas_ta
        h, l, c, v = self.data.High, self.data.Low, self.data.Close, self.data.Volume
        
        # Vortex Indicator
        vortex = pta.vortex(h, l, c, length=14)
        self.vi_plus = self.I(lambda: vortex['VORTl_14'], name='VI+')
        self.vi_minus = self.I(lambda: vortex['VORTs_14'], name='VI-')
        
        # Choppiness Index
        self.ci = self.I(pta.chop, h, l, c, length=14, name='CHOP')
        
        # ADX
        self.adx = self.I(talib.ADX, h, l, c, 14, name='ADX')
        
        # Volume MA
        self.volume_ma = self.I(talib.SMA, v, 20, name='Vol MA')
        
        # ATR for stops
        self.atr = self.I(talib.ATR, h, l, c, 14, name='ATR')
        
    def next(self):
        if not self.position:
            # Long entry logic 🌙🚀
            if ((self.vi_plus[-2] < self.vi_minus[-2] and self.vi_plus[-1] > self.vi_minus[-1]) and  # Bullish crossover
                (self.ci[-2] > 61.8 and self.ci[-1] < 61.8) and  # Crossunder 61.8
                self.adx[-1] > 25 and
                self.data.Volume[-1] > self.volume_ma[-1]):
                
                atr_val = self.atr[-1]
                risk_amount = self.equity * self.risk_percent
                position_size = int(round(risk_amount / (1.5 * atr_val)))
                
                if position_size > 0:
                    self.buy(
                        size=position_size,
                        sl=self.data.Close[-1] - 1.5*atr_val,
                        meta={'peak': self.data.High[-1]}
                    )
                    print(f"🌙🚀 MOON DEV LONG SIGNAL DETECTED! Entry: {self.data.Close[-1]:.2f}, Size: {position_size}, SL: {self.data.Close[-1]-1.5*atr_val:.2f}")
            
            # Short entry logic 🌙📉
            if ((self.vi_minus[-2] < self.vi_plus[-2] and self.vi_minus[-1] > self.vi_plus[-1]) and  # Bearish crossover
                (self.ci[-2] < 38.2 and self.ci[-1] > 38.2) and  # Crossover 38.2
                self.adx[-1] > 25 and
                self.data.Volume[-1] > self.volume_ma[-1]):
                
                atr_val = self.atr[-1]
                risk_amount = self.equity * self.risk_percent
                position_size = int(round(risk_amount / (1.5 * atr_val)))
                
                if position_size > 0:
                    self.sell(
                        size=position_size,
                        sl=self.data.Close[-1] + 1.5*atr_val,
                        meta={'trough': self.data.Low[-1]}
                    )
                    print(f"🌙📉 MOON DEV SHORT SIGNAL DETECTED! Entry: