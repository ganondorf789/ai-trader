Here's the complete fixed code with Moon Dev themed debug prints and all technical issues resolved:

```python
import pandas as pd
import pandas_ta as ta
import talib
from backtesting import Backtest, Strategy

class MomentumConverge(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    cmo_period = 14
    supertrend_period = 10
    supertrend_multiplier = 3.0
    vwap_dev_threshold = 1.5  # 1.5% deviation
    
    def init(self):
        # Clean data columns
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col.lower()])
        
        # Calculate indicators using self.I()
        self.cmo = self.I(talib.CMO, self.data.Close, self.cmo_period, name='CMO')
        
        # SuperTrend using pandas_ta
        supertrend = self.data.df.ta.supertrend(length=self.supertrend_period, multiplier=self.supertrend_multiplier)
        self.supertrend_line = self.I(lambda: supertrend[f'SUPERT_{self.supertrend_period}_{self.supertrend_multiplier}'], name='SuperTrend')
        self.supertrend_dir = self.I(lambda: supertrend[f'SUPERTd_{self.supertrend_period}_{self.supertrend_multiplier}'], name='SuperTrend_Direction')
        
        # VWAP and Deviation
        vwap = self.data.df.ta.vwap(high=self.data.High, low=self.data.Low, close=self.data.Close, volume=self.data.Volume)
        self.vwap = self.I(lambda: vwap, name='VWAP')
        self.vwap_dev = self.I(lambda: ((self.data.Close - vwap) / vwap) * 100, name='VWAP_Deviation')
        
        # ATR for Fibonacci extensions
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        
        print("🌙✨ MOON DEV INDICATORS INITIALIZED! Ready for launch! 🚀")

    def next(self):
        if len(self.data.Close) < 2:
            return
        
        current_close = self.data.Close[-1]
        current_cmo = self.cmo[-1]
        prev_cmo = self.cmo[-2]
        vwap_dev = self.vwap_dev[-1]
        supertrend_line = self.supertrend_line[-1]
        atr_val = self.atr[-1]
        
        # Moon Dev debug prints
        print(f"🌙 CMO: {current_cmo:.1f} | VWAP Dev: {vwap_dev:.2f}% | SuperTrend: {supertrend_line:.2f}")
        
        # Long Entry Conditions
        if (not self.position.is_long and
            (prev_cmo < 50 and current_cmo > 50) and  # CMO crossover 50
            (self.data.Close[-2] < self.supertrend_line[-2] and current_close > supertrend_line) and  # Price crossover SuperTrend
            vwap_dev >= self.vwap_dev_threshold):
            
            entry_price = current_close
            stop_loss = max(supertrend_line, entry_price - atr_val)
            risk_amount = self.equity * self.risk_per_trade
            risk_per_trade = entry_price - stop_loss
            
            if risk_per_trade <= 0:
                print("🌙✨⚠️ Risk per trade invalid. Aborting launch!")
                return
            
            position_size = int(round(risk_amount / risk_per_trade))
            if position_size == 0:
                return
                
            tp1 = entry_price + 1.618 * atr_val
            tp2 = entry_price + 2.618 * atr_val
            
            # Split into two positions
            size_tp1 = position_size // 2
            size_tp2 = position_size - size_tp1
            
            if size_tp1 > 0:
                self.buy(size=size_tp1, sl=stop_loss, tp=tp1, tag="MOON_DEV_LONG_TP1")
            if size_tp2 > 0:
                self.buy(size=size_tp2, sl=stop_loss, tp=tp2, tag="MOON_DEV_LONG_TP2")
            print