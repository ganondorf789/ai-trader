```python
# 🌙 Moon Dev's VortexFlow Reversal Backtest 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

class VortexFlowReversal(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    atr_multiplier = 1.5
    vi_period = 14
    volume_ma_period = 30
    inflow_threshold = 1.2  # 20% above average
    
    def init(self):
        # 🌟 Vortex Indicator Calculation
        shifted_low = self.I(lambda x: x.shift(1), self.data.Low)
        shifted_high = self.I(lambda x: x.shift(1), self.data.High)
        
        vm_plus = self.I(lambda h, l: h - l, self.data.High, shifted_low)
        vm_minus = self.I(lambda l, h: l - h, self.data.Low, shifted_high)
        
        sum_vm_plus = self.I(talib.SUM, vm_plus, timeperiod=self.vi_period)
        sum_vm_minus = self.I(talib.SUM, vm_minus, timeperiod=self.vi_period)
        
        tr = self.I(talib.TRANGE, self.data.High, self.data.Low, self.data.Close)
        sum_tr = self.I(talib.SUM, tr, timeperiod=self.vi_period)
        
        self.vi_plus = self.I(lambda a, b: a/b, sum_vm_plus, sum_tr)
        self.vi_minus = self.I(lambda a, b: a/b, sum_vm_minus, sum_tr)
        
        # 🌌 ATR Bands
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.upper_band = self.I(lambda c, a: c + a*self.atr_multiplier, self.data.Close, self.atr)
        self.lower_band = self.I(lambda c, a: c - a*self.atr_multiplier, self.data.Close, self.atr)
        
        # 📈 Volume Indicators
        self.inflow_ma = self.I(talib.SMA, self.data.inflow, self.volume_ma_period)
        self.outflow_ma = self.I(talib.SMA, self.data.outflow, self.volume_ma_period)
        self.volume_ma = self.I(talib.SMA, self.data.Volume, self.volume_ma_period)
        
    def next(self):
        # 🌑 Skip early bars without sufficient data
        if len(self.data) < max(self.vi_period, self.volume_ma_period) + 1:
            return
            
        # 🚫 Filter low-volume periods
        if self.data.Volume[-1] < 0.8 * self.volume_ma[-1]:
            print(f"🌙 Low volume ({self.data.Volume[-1]:.0f} < 80% MA), skipping...")
            return
            
        # 💹 Bullish Entry
        if crossover(self.vi_plus, self.vi_minus):
            if (self.data.inflow[-1] > self.inflow_threshold * self.inflow_ma[-1] and
                self.data.Close[-1] > self.lower_band[-1]):
                
                sl = self.lower_band[-1]
                risk = self.data.Close[-1] - sl
                if risk <= 0: return
                
                size = int(round((self.equity * self.risk_per_trade) / risk))
                tp = self.data.Close[-1] + 2*risk
                
                if size > 0:
                    self.buy(size=size, sl=sl, tp=tp)
                    print(f"🌙✨ BULLISH ENTRY @ {self.data.Close[-1]:.2f}")
                    print(f"   Size: {size}, SL: {sl:.2f}, TP: {tp:.2f} 🚀")
        
        # 🐻 Bearish Entry
        elif crossover(self.vi_minus, self.vi_plus):
            if (self.data.outflow[-1] > self.inflow_threshold * self.outflow_ma[-1] and
                self.data.Close[-1] < self.upper_band[-1]):
                
                sl = self.upper_band[-1]
                risk = sl - self.data.Close[-1]
                if risk <= 0: return
                
                size = int(round((self.equity * self.