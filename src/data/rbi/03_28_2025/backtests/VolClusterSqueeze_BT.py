```python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class VolClusterSqueeze(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    lookback_period = 100  # For volatility percentile calculation
    
    def init(self):
        # Calculate core indicators using TA-Lib
        self.upper_bb = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0], self.data.Close)
        self.lower_bb = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2], self.data.Close)
        self.mid_bb = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1], self.data.Close)
        self.rsi = self.I(talib.RSI, self.data.Close, 14)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.swing_high = self.I(talib.MAX, self.data.High, 20)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20)
        
        # Calculate Bollinger bandwidth
        self.bandwidth = self.I(lambda u, l, m: (u - l) / m, self.upper_bb, self.lower_bb, self.mid_bb)
        
    def next(self):
        if len(self.data) < self.lookback_period:
            return
        
        # Calculate volatility percentile dynamically
        current_bw = self.bandwidth[-1]
        historical_bw = self.bandwidth[-self.lookback_period:-1]
        bw_percentile = (sum(bw < current_bw for bw in historical_bw)/self.lookback_period)*100
        
        # Entry conditions 🌙
        vol_trigger = bw_percentile < 20
        price_cond = self.data.Close[-1] > 0.99 * self.upper_bb[-1]
        volume_spike = self.data.Volume[-1] > 1.5 * self.volume_sma[-1]
        
        if not self.position and vol_trigger and price_cond and volume_spike:
            # Risk management calculations 🚀
            entry_price = self.data.Close[-1]
            stop_price = self.swing_high[-1] + self.atr[-1]
            risk_amount = self.risk_per_trade * self.equity
            risk_per_unit = abs(stop_price - entry_price)
            
            if risk_per_unit > 0:
                position_size = int(round(risk_amount / risk_per_unit))
                if position_size > 0:
                    self.sell(size=position_size, stop=stop_price)
                    print(f"🌙 MOON DEV ENTRY 🌟 | Size: {position_size} | Entry: {entry_price:.2f} | SL: {stop_price:.2f}")
        
        # Exit conditions ✨
        if self.position:
            # Time-based exit
            if len(self.data) - self.position.entry_bar >= 3:
                self.position.close()
                print(f"🌙 MOON DEV TIME EXIT ⏳ | Bar: {len(self.data)}")
                return
            
            # RSI flip exit
            rsi_exit = self.rsi[-2] < 30 and self.rsi[-1] >= 30
            
            # Emergency exit
            emergency_exit = (self.data.Close[-2] > self.upper_bb[-2]) and (self.data.Close[-1] > self.upper_bb[-1])
            
            # Volatility expansion exit
            vol_exit = bw_percentile < 10
            
            if rsi_exit or emergency_exit or vol_exit:
                self.position.close()
                reason = "RSI Flip" if rsi_exit else "Emergency" if emergency_exit else "Vol Expansion"
                print(f"🌙 MOON DEV EXIT 🏁 | Reason: {reason}")

# Data handling 🌌
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if