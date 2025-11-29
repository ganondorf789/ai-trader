```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
import numpy as np

class SqueezeDivergence(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators
        # Bollinger Bands
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, name=['BB_Upper', 'BB_Mid', 'BB_Lower'])
        
        # VWAP and its MA
        self.vwap = self.I(ta.vwap, high=self.data.High, low=self.data.Low, close=self.data.Close, volume=self.data.Volume, name='VWAP')
        self.vwap_ma = self.I(talib.SMA, self.vwap, timeperiod=20, name='VWAP_MA')
        
        # Volume MA
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='Volume_MA')
        
        # ATR for exits
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR')
        
        # Bandwidth for squeeze detection
        self.bandwidth = self.I(lambda upper, lower: upper - lower, self.bb_upper, self.bb_lower, name='BB_Width')
        self.sma_bandwidth = self.I(talib.SMA, self.bandwidth, timeperiod=20, name='SMA_BB_Width')
        
        print("🌙 Moon Dev Indicators Initialized! ✨")

    def next(self):
        # Check for existing positions
        if self.position:
            return  # Exit early if in position
            
        # Current values
        price = self.data.Close[-1]
        vwap = self.vwap[-1]
        vwap_ma = self.vwap_ma[-1]
        volume = self.data.Volume[-1]
        volume_ma = self.volume_ma[-1]
        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]
        bandwidth = self.bandwidth[-1]
        sma_bandwidth = self.sma_bandwidth[-1]
        
        # Squeeze condition (bandwidth < 80% of SMA bandwidth)
        is_squeeze = bandwidth < sma_bandwidth * 0.8
        volume_ok = volume > volume_ma
        
        # Moon Dev Debug Prints
        print(f"🌙 Price: {price:.2f} | VWAP: {vwap:.2f} vs MA: {vwap_ma:.2f} | Volume: {volume:.2f} vs MA: {volume_ma:.2f}")
        print(f"🔍 BB Width: {bandwidth:.2f} vs SMA: {sma_bandwidth:.2f} | Squeeze: {'✅' if is_squeeze else '❌'}")
        
        # Long Entry Logic
        if (is_squeeze and
            price > bb_upper and
            crossover(self.vwap, self.vwap_ma) and
            volume_ok):
            
            risk_amount = self.equity * self.risk_percent
            atr_value = self.atr[-1]
            risk_per_share = 2 * atr_value  # 2x ATR stop
            position_size = int(round(risk_amount / risk_per_share))
            
            if position_size > 0:
                self.buy(size=position_size)
                print(f"🚀 MOON DEV LONG ENTRY! Size: {position_size} | TP: {price + 2*atr_value:.2f}")
        
        # Short Entry Logic
        elif (is_squeeze and
              price < bb_lower and
              crossunder(self.vwap, self.vwap_ma) and
              volume_ok):
            
            risk_amount = self.equity * self.risk_percent
            atr_value = self.atr[-1]
            risk_per_share = 2 * atr_value
            position_size