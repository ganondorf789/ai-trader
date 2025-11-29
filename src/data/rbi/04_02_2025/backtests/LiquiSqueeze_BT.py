```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION FOR LIQUISQUEEZE STRATEGY 🚀

# Required imports
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

class LiquiSqueezeStrategy(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_dev = 2
    squeeze_threshold = 0.005  # 0.5% of price
    volume_multiplier = 1.5
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌙 Clean data and handle columns
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col], inplace=True)
        
        # ✨ Calculate indicators using TA-Lib with self.I()
        # Bollinger Bands
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, timeperiod=self.bb_period, 
                              nbdevup=self.bb_dev, nbdevdn=self.bb_dev, matype=0, name='BB_UPPER')[0]
        self.bb_middle = self.I(talib.BBANDS, self.data.Close, timeperiod=self.bb_period,
                               nbdevup=self.bb_dev, nbdevdn=self.bb_dev, matype=0, name='BB_MID')[1]
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=self.bb_period,
                              nbdevup=self.bb_dev, nbdevdn=self.bb_dev, matype=0, name='BB_LOWER')[2]
        
        # Volatility Squeeze Metrics
        self.bb_width = self.I(lambda u, l: u - l, self.bb_upper, self.bb_lower, name='BB_WIDTH')
        self.bb_width_sma = self.I(talib.SMA, self.bb_width, timeperiod=self.bb_period, name='BB_WIDTH_SMA')
        
        # Liquidation Zones (Swing Highs/Lows)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='SWING_HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='SWING_LOW')
        
        # Volume Analysis
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VOL_SMA')
        
        # Parabolic SAR (Standard calculation)
        self.psar = self.I(talib.SAR, self.data.High, self.data.Low, 
                          acceleration=0.02, maximum=0.2, name='PSAR')
        
    def next(self):
        # 🌑 Avoid weekends (crude low-liquidity filter)
        if self.data.index[-1].weekday() >= 5:
            return
        
        # 🌙✨ Core Strategy Logic
        price = self.data.Close[-1]
        swing_high = self.swing_high[-1]
        swing_low = self.swing_low[-1]
        
        # ===== ENTRY LOGIC =====
        if not self.position:
            # Long Entry Conditions
            long_cond = (
                (price >= swing_low * 0.995) and  # 0.5% below swing low
                (self.bb_width_sma[-1] < self.squeeze_threshold * price) and
                (price > self.bb_upper[-1]) and
                (self.data.Volume[-1] > self.volume_multiplier * self.volume_sma[-1])
            
            # Short Entry Conditions
            short_cond = (
                (price <= swing_high * 1.005) and  # 0.5% above swing high
                (self.bb_width_sma[-1] < self.squeeze_threshold * price) and
                (price < self.bb_lower[-1]) and
                (self.data.Volume[-1] > self.volume_multiplier * self.volume_sma[-1])
            
            # 🚀 Execute entries with risk management
            if long_cond:
                stop_price = min(self.bb_lower[-1], swing_low)
                risk_amount = self.equity * self.risk_percent
                position_size = int(round(risk_amount / (price - stop_price)))
                if position_size > 0:
                    self.buy(size