Here's the fixed code with all backtesting.lib imports removed and proper replacements implemented:

```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR LIQUIDATIONVOLT STRATEGY 🚀

# Required imports
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

class LiquidationVoltStrategy(Strategy):
    # Strategy parameters
    risk_per_trade = 0.01  # 1% risk per trade
    bb_period = 20
    bb_dev = 2
    atr_period = 14
    swing_period = 20
    liquidation_lookback = 2880  # 30 days in 15m intervals
    
    def init(self):
        # 🌙 Data Preparation
        self.data.df.index = pd.to_datetime(self.data.df.index)
        
        # 📊 Indicator Calculations
        # Bollinger Bands
        close = self.data.Close
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(talib.BBANDS, close, 
                                                              timeperiod=self.bb_period,
                                                              nbdevup=self.bb_dev,
                                                              nbdevdn=self.bb_dev,
                                                              matype=0, name='BBANDS')
        
        # Liquidation Analysis
        self.liquidation = self.data['Liquidation']
        self.liquidation_avg = self.I(talib.SMA, self.liquidation,
                                     timeperiod=self.liquidation_lookback,
                                     name='Liquidation_30d_MA')
        
        # Volatility-Sensitive Volume Oscillator
        high, low, close, volume = self.data.High, self.data.Low, self.data.Close, self.data.Volume
        ad = self.I(talib.AD, high, low, close, volume, name='AD')
        ad_ma = self.I(talib.SMA, ad, timeperiod=20, name='AD_MA')
        self.vol_osc = ad - ad_ma  # Custom volatility oscillator
        
        # ATR for Risk Management
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')
        
        # Swing Highs/Lows
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_period, name='Swing_High')
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_period, name='Swing_Low')
        
        # Volatility Filter
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=self.liquidation_lookback, name='ATR_MA')
        
    def next(self):
        # 🌙 Market Volatility Filter
        if self.atr[-1] > 1.5 * self.atr_ma[-1]:
            print("🌧️ MOON DEV ALERT: High volatility detected - skipping trades")
            return  # Skip trades in high volatility
        
        # 🚀 Entry Logic
        price = self.data.Close[-1]
        upper_band = self.bb_upper[-1]
        lower_band = self.bb_lower[-1]
        
        # Liquidation Cluster Check
        liquidation_trigger = (self.liquidation[-1] > 2 * self.liquidation_avg[-1])
        
        # Bullish Reversal Setup
        bullish_cross = (self.vol_osc[-2] < 0 and self.vol_osc[-1] > 0)
        if (price <= lower_band * 1.05 and liquidation_trigger and
            self.vol_osc[-1] > self.vol_osc[-2] and
            bullish_cross):
            
            # Risk Management
            sl_price = min(self.swing_low[-1], lower_band) - self.atr[-1]
            risk_amount = self.equity * self.risk_per_trade
            position_size = int(round(risk_amount / (price - sl_price)))
            
            if position_size > 0:
                self.buy(size=position_size, sl=sl_price,
                        tp=self.bb_middle[-1],
                        tag="🌙 BULLISH REVERSAL")
                print(f"🚀 MOON DEV LONG ENTRY 🚀 | Price: {price:.2f} | Size: {position_size} | "
                      f"SL: {sl_price:.2f} | TP: {self.bb_middle[-1]:.2f}")
        
        # Bearish Reversal Setup
        bearish_cross = (self.vol_osc[-2] >