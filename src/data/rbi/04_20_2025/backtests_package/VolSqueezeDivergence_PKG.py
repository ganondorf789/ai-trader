Here's the fixed code with all backtesting.lib references removed and proper indicator implementations:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import numpy as np

class VolSqueezeDivergence(Strategy):
    bb_period = 20
    bb_dev = 2
    bandwidth_lookback = 120
    put_call_sma_period = 20
    volume_percentile = 0.9
    atr_period = 14
    risk_pct = 0.02
    exit_bars = 15

    def init(self):
        close = self.data.Close
        self.upper, self.middle, self.lower = self.I(talib.BBANDS, close, 
                                                   timeperiod=self.bb_period,
                                                   nbdevup=self.bb_dev,
                                                   nbdevdn=self.bb_dev)
        
        self.bandwidth = self.I(lambda u, l, m: (u - l) / m,
                               self.upper, self.lower, self.middle)
        self.bandwidth_min = self.I(talib.MIN, self.bandwidth, self.bandwidth_lookback)
        
        self.put_call_ratio = self.data['Put_Call_Ratio']
        self.put_call_sma = self.I(talib.SMA, self.put_call_ratio, self.put_call_sma_period)
        
        self.upside_volume = self.I(lambda c, v: np.where(c > np.roll(c, 1), v, 0),
                             self.data.Close, self.data.Volume)
        self.upside_90 = self.I(lambda x: x.rolling(int(self.bb_period*2)).quantile(self.volume_percentile),
                          self.upside_volume)
        
        self.downside_volume = self.I(lambda c, v: np.where(c < np.roll(c, 1), v, 0),
                              self.data.Close, self.data.Volume)
        self.downside_90 = self.I(lambda x: x.rolling(int(self.bb_period*2)).quantile(self.volume_percentile),
                           self.downside_volume)
        
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, close, self.atr_period)

    def next(self):
        if len(self.data) < max(self.bandwidth_lookback, self.bb_period) + 1:
            return

        current_close = self.data.Close[-1]
        bandwidth = self.bandwidth[-1]
        bandwidth_min = self.bandwidth_min[-1]

        put_call = self.put_call_ratio[-1]
        put_call_sma = self.put_call_sma[-1]

        volume = self.data.Volume[-1]
        upside_vol = self.upside_volume[-1]
        downside_vol = self.downside_volume[-1]

        if not self.position:
            if (bandwidth <= bandwidth_min and
                put_call < put_call_sma * 0.85 and
                upside_vol >= self.upside_90[-1] and
                current_close > self.upper[-1]):
                
                stop_loss = self.lower[-1]
                risk_amount = self.risk_pct * self.equity
                risk_per_trade = current_close - stop_loss
                
                if risk_per_trade > 0:
                    position_size = int(round(risk_amount / risk_per_trade))
                    if position_size > 0:
                        self.buy(size=position_size, sl=stop_loss)
                        print(f"🌙✨ MOON DEV LONG ENTRY: {current_close} 🚀 PCR: {put_call:.2f} | Vol: {volume:.2f}")

            elif (bandwidth <= bandwidth_min and
                  put_call > put_call_sma * 1.15 and
                  downside_vol >= self.downside_90[-1] and
                  current_close < self.lower[-1]):
                  
                  stop_loss = self.upper[-1]
                  risk_amount = self.risk_pct * self.equity
                  risk_per_trade = stop_loss - current_close
                  
                  if risk_per_trade > 0:
                      position_size = int(round(risk_amount / risk_per_trade))
                      if position_size > 0:
                          self.sell(size=position_size, sl=stop_loss)
                          print(f"🌙✨ MOON DEV SHORT ENTRY: {current_close} 🚀 PCR: {put_call:.2f} | Vol: {volume:.2f}")

        else:
            for trade in self.trades:
                if trade.is_long:
                    if current_close < self.middle[-1]: