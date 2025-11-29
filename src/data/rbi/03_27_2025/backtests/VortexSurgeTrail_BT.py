```python
# 🌙 Moon Dev's VortexSurgeTrail Backtest 🌙
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib
import numpy as np

# Clean data columns & handle formatting
def load_data(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

# Vortex Indicator helper functions
def vortex_plus(high, low, close, timeperiod):
    vi_p, _ = talib.VORTEX(high, low, close, timeperiod)
    return vi_p

def vortex_minus(high, low, close, timeperiod):
    _, vi_m = talib.VORTEX(high, low, close, timeperiod)
    return vi_m

class VortexSurgeTrail(Strategy):
    risk_per_trade = 0.01  # 1% of equity per trade
    
    def init(self):
        # 🌙 Vortex Indicator
        self.vi_plus = self.I(vortex_plus, self.data.High, self.data.Low, self.data.Close, 14, name='VI+')
        self.vi_minus = self.I(vortex_minus, self.data.High, self.data.Low, self.data.Close, 14, name='VI-')
        
        # ✨ Chande Momentum Oscillator
        self.cmo = self.I(talib.CMO, self.data.Close, 14, name='CMO')
        
        # 🌀 Keltner Channel Components
        self.ema = self.I(talib.EMA, self.data.Close, 20, name='EMA')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 20, name='ATR')
        self.upper_keltner = self.I(lambda ema, atr: ema + 2*atr, self.ema, self.atr, name='Upper Keltner')
        self.lower_keltner = self.I(lambda ema, atr: ema - 2*atr, self.ema, self.atr, name='Lower Keltner')
        self.keltner_width = self.I(lambda u, l: u - l, self.upper_keltner, self.lower_keltner, name='Keltner Width')
        self.avg_width = self.I(talib.SMA, self.keltner_width, 20, name='Avg Width')
        
        # 🏔️ Swing High/Low
        self.swing_high = self.I(talib.MAX, self.data.High, 20, name='Swing High')
        self.swing_low = self.I(talib.MIN, self.data.Low, 20, name='Swing Low')

    def next(self):
        # 🌑 Avoid low volatility periods
        if self.keltner_width[-1] <= 1.5 * self.avg_width[-1]:
            print("🌙 Volatility too low - cosmic energies calm")
            return
            
        # 🚀 Long Entry Sequence
        if not self.position.is_long and \
           crossover(self.vi_plus, self.vi_minus) and \
           self.cmo[-1] > 60 and \
           self.data.Close[-1] > self.swing_high[-1]:
            
            risk_amount = self.equity * self.risk_per_trade
            entry_price = self.data.Close[-1]
            stop_price = self.lower_keltner[-1]
            risk_per_share = entry_price - stop_price
            
            if risk_per_share <= 0:
                print(f"🚫 Invalid cosmic alignment (risk: {risk_per_share:.2f})")
                return
                
            size = int(round(risk_amount / risk_per_share))
            self.buy(size=size, sl=stop_price, tag='BullishSurge')
            print(f"🚀🌙 LONG ENTRY! Size: {size} | Entry: {entry_price:.2f} | Cosmic Stop: {stop_price:.2f}")

        # 🌑 Short Entry Sequence
        if not self.position.is_short and \
           crossover(self.vi_minus, self.vi_plus) and \
           self.cmo[-1] < -