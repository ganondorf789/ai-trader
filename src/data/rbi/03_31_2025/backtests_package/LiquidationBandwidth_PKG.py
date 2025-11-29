Here's the fixed code with all Moon Dev package improvements and proper indicator handling:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import numpy as np  # 🌙 Added for array operations

# 🌙 Moon Dev Data Preparation Magic ✨
def prepare_data(path):
    data = pd.read_csv(path)
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
    data.set_index('datetime', inplace=True)
    return data

class LiquidationBandwidth(Strategy):
    def init(self):
        # 🌙 Liquidation Cluster Indicators (Pure TA-Lib Implementation)
        self.lh = self.I(talib.MAX, self.data.High, timeperiod=50, name='Liquidation High')
        self.ll = self.I(talib.MIN, self.data.Low, timeperiod=50, name='Liquidation Low')

        # ✨ Bollinger Bandwidth Setup (TA-Lib + Custom Calculation)
        upper, mid, lower = self.I(talib.BBANDS, self.data.Close, 
                                  timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        self.bbw = self.I(lambda u, l, m: (u-l)/m, upper, lower, mid, name='BB Width')
        self.bbw_ma = self.I(talib.SMA, self.bbw, timeperiod=20, name='BBW MA')

        # 📈 Volatility Calculations (Pure TA-Lib)
        self.hv = self.I(talib.STDDEV, self.data.Close, timeperiod=14, name='14D HV')
        self.hv_90 = self.I(lambda x: pd.Series(x).rolling(100).quantile(0.9).values, 
                          self.hv, name='HV 90th %ile')

        # 🌙 Daily Loss Tracking
        self.last_date = None
        self.daily_start_equity = None

    def next(self):
        # 💸 Risk Management First!
        current_date = self.data.index[-1].date()
        if self.last_date != current_date:
            self.last_date = current_date
            self.daily_start_equity = self.equity
            print(f"🌙 New Trading Day: {current_date} | Starting Equity: ${self.daily_start_equity:,.2f}")

        if (self.equity/self.daily_start_equity - 1) <= -0.03:
            print(f"🌑💥 EMERGENCY STOP! Daily Loss Limit Hit at {self.equity:,.2f}")
            self.position.close()
            return

        # 📉 Exit Conditions (Pure Array Indexing)
        if self.position and self.hv[-1] > self.hv_90[-1]:
            print(f"🌙✨ HV Exit Signal ({self.hv[-1]:.2f} > {self.hv_90[-1]:.2f})")
            self.position.close()
            return

        # 🚀 Entry Logic (Long Example - No backtesting.lib used)
        if not self.position and len(self.data) > 50:
            # Cluster Approach Check (Array Indexing)
            cluster_high = self.lh[-1]
            price_approach = (self.data.Close[-1] >= cluster_high * 0.985 and 
                             self.data.Close[-1] <= cluster_high)
            
            # Volatility Contraction Check (Array Comparison)
            bb_contraction = self.bbw[-1] < self.bbw_ma[-1]
            
            # Breakout Confirmation (Direct Comparison)
            breakout = self.data.Close[-1] > cluster_high

            if price_approach and bb_contraction and breakout:
                # 🌙 Risk Calculation
                risk_amount = self.equity * 0.01
                stop_loss = self.ll[-1]  # Cluster low as SL
                risk_per_unit = self.data.Close[-1] - stop_loss
                
                if risk_per_unit > 0:
                    size = int(round(risk_amount / risk_per_unit))
                    take_profit = self.data.Close[-1] + 2*risk_per_unit
                    
                    print(f"🚀🌙 LONG Signal | Size: {size} | Entry: {self.data.Close[-1]:.2f}")
                    print