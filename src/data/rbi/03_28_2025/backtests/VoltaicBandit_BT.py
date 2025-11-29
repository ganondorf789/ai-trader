```python
# 🌙 Moon Dev's VoltaicBandit Backtest Implementation ✨

import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

class VoltaicBandit(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    max_positions = 3    # Maximum concurrent positions
    
    def init(self):
        # 🌙 Calculate all required indicators using proper TA-Lib functions
        self.vwap = self.I(ta.vwap, 
                          high=self.data.High, low=self.data.Low,
                          close=self.data.Close, volume=self.data.Volume,
                          name='VWAP')
        
        # Volatility components
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.std_dev = self.I(talib.STDDEV, self.data.Close, 20)
        
        # Dynamic bands (VWAP ± 2*STDDEV)
        self.upper_band = self.I(lambda: self.vwap + 2*self.std_dev, name='UPPER_BAND')
        self.lower_band = self.I(lambda: self.vwap - 2*self.std_dev, name='LOWER_BAND')
        
        # Volume confirmation
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20)
        
        # Momentum indicators
        self.rsi = self.I(talib.RSI, self.data.Close, 14)
        self.ema50 = self.I(talib.EMA, self.data.Close, 50)
        
        print("🌙 VoltaicBandit initialized with Moon Dev precision! 🚀")

    def next(self):
        # 🌟 Moon Dev debug prints
        price = self.data.Close[-1]
        # print(f"🌙 Processing candle {len(self.data)-1} | Price: {price:.2f}")
        
        # Skip if max positions reached
        if len(self.positions) >= self.max_positions:
            return

        # Current market conditions
        band_width = self.upper_band[-1] - self.lower_band[-1]
        atr_value = self.atr[-1]
        volume_spike = self.data.Volume[-1] > 1.2 * self.volume_ma[-1]
        trend_filter = self.ema50[-1]

        # 🌙 Entry Logic
        if band_width < 1.5 * atr_value and volume_spike:
            # Long entry conditions
            if (price > self.upper_band[-1] and price > trend_filter and
                not self.position.is_long):
                
                # Risk management calculations
                stop_loss = price - 2 * atr_value
                risk_amount = self.equity * self.risk_percent
                position_size = int(round(risk_amount / (price - stop_loss)))
                
                if position_size > 0:
                    self.buy(size=position_size, 
                            sl=stop_loss,
                            tp=price + 4*atr_value)
                    print(f"🚀🌙 LONG ENTRY | Size: {position_size} | Entry: {price:.2f}")
            
            # Short entry conditions
            elif (price < self.lower_band[-1] and price < trend_filter and
                  not self.position.is_short):
                
                stop_loss = price + 2 * atr_value
                risk_amount = self.equity * self.risk_percent
                position_size = int(round(risk_amount / (stop_loss - price)))
                
                if position_size > 0:
                    self.sell(size=position_size,
                             sl=stop_loss,
                             tp=price - 4*atr_value)
                    print(f"🌑🌙 SHORT ENTRY | Size: {position_size} | Entry: {price:.2f}")

        # 🌙 Exit Logic
        for trade in self.positions:
            if trade.is_long:
                # RSI divergence check
                if (self.data.High[-1] > self.data.High[-2] and
                    self.rsi[-1] < self.rsi[-2]):
                    trade.close()
                    print(f"⚠️🌙 LONG EXIT (RSI Divergence) | Price: {price:.2f}")
                
                # Band re-entry check
                elif price < self.upper_band[-1]:
                    trade.close()
                    print(f"🎯🌙 LONG EXIT (Band Re-entry) | Price: {price:.2f}")
            
            elif trade.is_short:
                # RSI divergence check
                if (self