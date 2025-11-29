Here's the fixed and completed backtest code with Moon Dev themed debug prints and proper position sizing:

```python
# 🌙 Moon Dev's Volatility Squeeze Reversal Backtest 🌙
from backtesting import Backtest, Strategy
import talib
import pandas as pd
import numpy as np

class VolatilitySqueezeReversal(Strategy):
    def init(self):
        # 🌙 Strategy Parameters
        self.bb_period = 20
        self.bb_dev = 2
        self.contraction_period = 14
        self.funding_lookback = 96  # 24h in 15m intervals
        self.risk_percent = 0.01
        
        # 🌙 Indicator Calculation
        # Bollinger Bands
        self.upper_band = self.I(lambda close: talib.BBANDS(close, timeperiod=self.bb_period, nbdevup=self.bb_dev, nbdevdn=self.bb_dev)[0], self.data.Close)
        self.middle_band = self.I(lambda close: talib.BBANDS(close, timeperiod=self.bb_period, nbdevup=self.bb_dev, nbdevdn=self.bb_dev)[1], self.data.Close)
        self.lower_band = self.I(lambda close: talib.BBANDS(close, timeperiod=self.bb_period, nbdevup=self.bb_dev, nbdevdn=self.bb_dev)[2], self.data.Close)
        
        # Bollinger Band Width
        self.bb_width = self.I(lambda u, l, m: (u - l)/m, self.upper_band, self.lower_band, self.middle_band)
        
        # Volatility Contraction
        self.bb_width_min = self.I(talib.MIN, self.bb_width, timeperiod=self.contraction_period)
        self.bb_width_max = self.I(talib.MAX, self.bb_width, timeperiod=self.contraction_period)
        
        # Funding Rate Extremes
        self.funding_max = self.I(talib.MAX, self.data.Funding_Rate, timeperiod=self.funding_lookback)
        self.funding_min = self.I(talib.MIN, self.data.Funding_Rate, timeperiod=self.funding_lookback)
        
    def next(self):
        # 🌙 Avoid Overlapping Trades
        if self.position:
            return
            
        # 🌙 Current Values
        price = self.data.Close[-1]
        upper = self.upper_band[-1]
        lower = self.lower_band[-1]
        middle = self.middle_band[-1]
        funding = self.data.Funding_Rate[-1]
        
        # 🌙 Volatility Contraction Check
        contraction_threshold = self.bb_width_min[-1] + 0.2*(self.bb_width_max[-1] - self.bb_width_min[-1])
        is_contracted = self.bb_width[-1] < contraction_threshold
        
        # 🌙 Funding Rate Extremes
        funding_90th = self.funding_min[-1] + 0.9*(self.funding_max[-1] - self.funding_min[-1])
        funding_10th = self.funding_min[-1] + 0.1*(self.funding_max[-1] - self.funding_min[-1])
        
        # 🌙 Entry Logic
        if is_contracted:
            # Long Entry 🚀
            if funding >= funding_90th and price > upper:
                sl = lower
                risk_per_share = price - sl
                if risk_per_share <= 0:
                    print("🌙✨ Invalid SL for Long, skipping entry")
                    return
                position_size = int(round((self.equity * self.risk_percent) / risk_per_share))
                if position_size > 0:
                    print(f"🚀🌙 LONG SIGNAL! Price: {price:.2f} | Size: {position_size}")
                    self.buy(size=position_size, sl=sl)
            
            # Short Entry 🌑
            elif funding <= funding_10th and price < lower:
                sl = upper
                risk_per_share = sl - price
                if risk_per_share <= 0:
                    print("🌙✨ Invalid SL for Short, skipping entry")
                    return
                position_size = int(round((self.equity * self.risk_percent) / risk_per_share))
                if position_size > 0:
                    print(f"🌑🌙 SHORT SIGNAL! Price: {price:.2f} | Size: {position_size}")
                    self.sell(size=position_size, sl=sl)
                    
        # 🌙 Exit Logic
        for trade in self.trades:
            if