```python
# 🌙 Moon Dev's VolSurge Bandit Backtest 🌙
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import datetime

class VolSurgeBandit(Strategy):
    def init(self):
        # 🚀 Initialize Indicators with Moon Dev Magic
        self.vwap = self.I(ta.vwap, self.data.High, self.data.Low, self.data.Close, self.data.Volume, name='VWAP')
        self.atr = self.I(ta.atr, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        self.vwap_upper = self.I(lambda vwap, atr: vwap + 2*atr, self.vwap, self.atr, name='UPPER_BAND')
        self.vwap_lower = self.I(lambda vwap, atr: vwap - 2*atr, self.vwap, self.atr, name='LOWER_BAND')
        self.vol_ma = self.I(ta.sma, self.data.Volume, 20, name='VOL_MA')
        
        # 🌙 Track equity for dynamic position sizing
        self.equity = self._broker.starting_cash
        print("✨ Moon Dev Indicators Activated! ✨")

    def next(self):
        # 🌙 Update current equity
        self.equity = self._broker.equity
        
        # 🛑 Skip early bars without sufficient data
        if len(self.data) < 20:
            return
            
        # ⏳ Time filter: Avoid first/last 30 minutes
        current_time = self.data.index[-1].time()
        if (current_time <= datetime.time(0, 30)) or (current_time >= datetime.time(23, 30)):
            return

        # 🚀 LONG Entry Logic
        if not self.position:
            if (self.data.Close[-2] > self.vwap_upper[-2] and
                self.data.Volume[-2] > 1.5 * self.vol_ma[-2] and
                self.data.Open[-1] > self.data.Close[-2]):
                
                # 🌙 Risk Management Calculation
                stop_loss = self.vwap_lower[-2]
                risk_per_share = self.data.Open[-1] - stop_loss
                if risk_per_share > 0:
                    risk_amount = self.equity * 0.01  # 1% risk
                    position_size = int(round(risk_amount / risk_per_share))
                    take_profit = self.data.Open[-1] + 2*risk_per_share
                    
                    self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                    print(f"🌕🚀 LONG SIGNAL! Size: {position_size} | Entry: {self.data.Open[-1]:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")

        # 🌑 SHORT Entry Logic
        if not self.position:
            if (self.data.Close[-2] < self.vwap_lower[-2] and
                self.data.Volume[-2] > 1.5 * self.vol_ma[-2] and
                self.data.Open[-1] < self.data.Close[-2]):
                
                # 🌙 Risk Management Calculation
                stop_loss = self.vwap_upper[-2]
                risk_per_share = stop_loss - self.data.Open[-1]
                if risk_per_share > 0:
                    risk_amount = self.equity * 0.01  # 1% risk
                    position_size = int(round(risk_amount / risk_per_share))
                    take_profit = self.data.Open[-1] - 2*risk_per_share
                    
                    self.sell(size=position_size, sl=stop_loss, tp=take_profit)
                    print(f"🌑✨ SHORT SIGNAL! Size: {position_size} | Entry: {self.data.Open[-1]:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")

        # 💎 Exit Logic
        if self.position.is_long and crossover(self.vwap[-1], self.data.Close[-1]):
            self.position.close()
            print(f"🌙📉 LONG EXIT at VWAP {self.vwap[-1]:.2f}")
            
        if self.position.is_short and crossover(self.data.Close[-1], self.vwap[-1]):
            self.position.close()
            print(f"🌙📈 SHORT EXIT at VWAP {self.vwap[-1]:.2f}")

# 🧹