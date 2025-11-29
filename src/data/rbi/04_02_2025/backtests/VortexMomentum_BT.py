```python
# 🌙 Moon Dev's VortexMomentum Backtest 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class VortexMomentum(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    atr_multiplier = 1.5
    cmf_threshold = 0.1
    
    def init(self):
        # 🌟 Indicator Calculation Phase 🌟
        self.vi_plus, self.vi_minus = self.calculate_vortex()
        self.cmf = self.I(talib.CMF, self.data.High, self.data.Low, self.data.Close, self.data.Volume, 20, name='CMF')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20, name='Volume_MA')
        
    def calculate_vortex(self):
        # 🌀 Vortex Indicator Calculation 🌀
        high = self.data.High
        low = self.data.Low
        
        # Calculate True Range
        tr = talib.TRANGE(high, low, self.data.Close)
        
        # Calculate VM+ and VM-
        vm_plus = (high - low.shift(1)).fillna(0)
        vm_minus = (high.shift(1) - low).fillna(0)
        
        # 14-period sums
        sum_vm_plus = talib.SUM(vm_plus, 14)
        sum_vm_minus = talib.SUM(vm_minus, 14)
        sum_tr = talib.SUM(tr, 14)
        
        # Avoid division by zero
        sum_tr = sum_tr.replace(0, 1e-10)
        
        vi_plus = sum_vm_plus / sum_tr
        vi_minus = sum_vm_minus / sum_tr
        
        return (
            self.I(lambda x: x, vi_plus, name='VI+'),
            self.I(lambda x: x, vi_minus, name='VI-')
        )

    def next(self):
        # 🚀 Moon Dev Trading Logic 🚀
        if self.position:
            return  # Exit check takes priority
            
        # Volume filter 🌊
        if self.data.Volume[-1] < self.volume_ma[-1]:
            return
            
        # Long Entry Conditions 🌈
        if (crossover(self.vi_plus, self.vi_minus) and
            self.cmf[-1] > 0 and
            self.data.Low[-1] < self.data.Low[-2] and
            self.cmf[-1] > self.cmf[-2]):
            
            sl_price = self.data.Close[-1] - self.atr[-1] * self.atr_multiplier
            self.calculate_position(sl_price, direction='long')

        # Short Entry Conditions 🌧️    
        elif (crossover(self.vi_minus, self.vi_plus) and
              self.cmf[-1] < 0 and
              self.data.High[-1] > self.data.High[-2] and
              self.cmf[-1] < self.cmf[-2]):
            
            sl_price = self.data.Close[-1] + self.atr[-1] * self.atr_multiplier
            self.calculate_position(sl_price, direction='short')

    def calculate_position(self, sl_price, direction):
        # 💰 Risk Management Magic 💰
        risk_per_share = abs(self.data.Close[-1] - sl_price)
        if risk_per_share == 0: return
        
        position_size = (self.equity * self.risk_per_trade) / risk_per_share
        position_size = int(round(position_size))
        
        if position_size == 0: return
        
        if direction == 'long':
            self.buy(size=position_size, sl=sl_price)
            print(f"🌙✨ LONG Signal Activated! Size: {position_size} | Entry: {self.data.Close[-1]:.2f} | SL: {sl_price:.2f}")
        else:
            self.sell(size=position_size, sl=sl_price)
            print(f"🌙🌧️ SHORT Signal Activated! Size: {position_size} | Entry: {self.data.Close[-1]:.2f} | SL: {sl_price:.2f}")

# Data Loading & Cleaning 🌐
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-t