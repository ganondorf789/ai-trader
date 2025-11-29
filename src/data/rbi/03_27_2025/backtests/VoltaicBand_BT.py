```python
# 🌙 Moon Dev's VoltaicBand Backtest Implementation
import pandas as pd
import talib
import pandas_ta as pta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class VoltaicBand(Strategy):
    # Strategy parameters
    atr_period = 14
    vwap_period = 20
    atr_multiplier = 2
    risk_pct = 0.02
    volatility_threshold = 0.1  # 10% of VWAP
    
    def init(self):
        # 🌙 Indicator Calculation
        self.vwap = self.I(self._calculate_vwap, self.data.High, self.data.Low, 
                          self.data.Close, self.data.Volume, self.vwap_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, 
                         self.data.Close, self.atr_period)
        self.upper_band = self.I(lambda v,a: v + self.atr_multiplier*a, 
                                self.vwap, self.atr)
        self.lower_band = self.I(lambda v,a: v - self.atr_multiplier*a, 
                                self.vwap, self.atr)
        
    def _calculate_vwap(self, high, low, close, volume, period):
        """✨ Moon Dev's Special VWAP Calculation"""
        return pta.vwap(high=high, low=low, close=close, volume=volume, length=period)
        
    def next(self):
        # 🌙 Volatility Filter Check
        current_vwap = self.vwap[-1]
        current_atr = self.atr[-1]
        
        if current_atr < (current_vwap * self.volatility_threshold):
            print(f"🌙 Volatility Too Low! ATR({current_atr:.2f}) < 10% VWAP({current_vwap*0.1:.2f})")
            return
            
        # 🚀 Entry Logic
        if not self.position:
            # Long Entry
            if self.data.Close[-1] > self.upper_band[-1]:
                risk_amount = self.risk_pct * self.equity
                position_size = int(round(risk_amount / current_atr))
                if position_size > 0:
                    self.buy(size=position_size, 
                            sl=self.data.Close[-1] - current_atr,
                            tag='LONG')
                    print(f"🚀 LONG | Size: {position_size} | Entry: {self.data.Close[-1]:.2f}")
            
            # Short Entry
            elif self.data.Close[-1] < self.lower_band[-1]:
                risk_amount = self.risk_pct * self.equity
                position_size = int(round(risk_amount / current_atr))
                if position_size > 0:
                    self.sell(size=position_size, 
                             sl=self.data.Close[-1] + current_atr,
                             tag='SHORT')
                    print(f"🌑 SHORT | Size: {position_size} | Entry: {self.data.Close[-1]:.2f}")
        
        # ✨ Exit Logic
        else:
            entry_price = self.position.entry_price
            bars_held = len(self) - self.position.entry_bar
            
            if self.position.is_long:
                # Track highest high since entry
                high_since_entry = max(self.data.High[self.position.entry_bar:-1])
                retrace_level = (entry_price + high_since_entry) / 2
                
                if self.data.Low[-1] <= retrace_level:
                    self.position.close()
                    print(f"✨ LONG EXIT | Profit: {self.data.Close[-1]-entry_price:.2f}")
                elif bars_held >= 5:
                    self.position.close()
                    print(f"⏳ LONG TIME EXIT | Bars Held: {bars_held}")
                    
            elif self.position.is_short:
                # Track lowest low since entry
                low_since_entry = min(self.data.Low[self.position.entry_bar:-1])
                retrace_level = (entry_price + low_since_entry) / 2
                
                if self.data.High[-1] >= retrace_level:
                    self.position.close()
                    print(f"✨ SHORT EXIT | Profit: {entry_price-self.data.Close[-1]:.2f}")
                elif bars_held >= 5:
                    self.position.close()
                    print(f"⏳ SHORT TIME EXIT | Bars Held: {bars_held}")

# 🌙 Data Preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m