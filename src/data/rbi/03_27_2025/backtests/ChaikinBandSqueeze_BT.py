```python
# 🌙✨ Moon Dev's ChaikinBandSqueeze Backtest 🌙🚀

import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Data Preparation
def load_data(path):
    data = pd.read_csv(path)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Drop unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    # Ensure proper column mapping
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    return data

class ChaikinBandSqueeze(Strategy):
    risk_percent = 1  # 1% risk per trade 🌙
    atr_period = 14
    swing_period = 20
    bb_period = 20
    bb_dev = 2
    chaikin_fast = 3
    chaikin_slow = 10
    chaikin_sma_period = 50
    
    def init(self):
        # 🌙✨ Indicator Calculations
        # Chaikin Oscillator
        self.chaikin = self.I(talib.ADOSC, self.data.High, self.data.Low, self.data.Close, self.data.Volume,
                              fastperiod=self.chaikin_fast, slowperiod=self.chaikin_slow, name='Chaikin')
        
        # Chaikin SMA
        self.chaikin_sma = self.I(talib.SMA, self.chaikin, timeperiod=self.chaikin_sma_period, name='Chaikin_SMA')
        
        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(talib.BBANDS, self.data.Close, 
                                                              timeperiod=self.bb_period, nbdevup=self.bb_dev, 
                                                              nbdevdn=self.bb_dev, matype=0, name=['BB_Upper', 'BB_Middle', 'BB_Lower'])
        
        # BB Width
        self.bb_width = (self.bb_upper - self.bb_lower) / self.bb_middle
        self.bb_width = self.I(lambda: self.bb_width, name='BB_Width')
        
        # BB Width Percentile (20-period 20th percentile)
        self.bb_width_percentile = self.I(lambda x: x.rolling(20).quantile(0.2), self.bb_width, name='BB_Width_Pctl')
        
        # Swing High/Low
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period, name='Swing_High')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period, name='Swing_Low')
        
        # ATR for trailing stop
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period, name='ATR')
        
        print("🌙✨ Strategy Initialized with Moon Power! 🌙🚀")

    def next(self):
        # Skip bars with incomplete data
        if np.isnan(self.chaikin[-1]) or np.isnan(self.chaikin_sma[-1]) or \
           np.isnan(self.bb_width[-1]) or np.isnan(self.bb_width_percentile[-1]):
            return

        # 🌙✨ Entry Conditions
        bb_squeeze = self.bb_width[-1] < self.bb_width_percentile[-1]
        chaikin_above = crossover(self.chaikin, self.chaikin_sma)
        chaikin_below = crossunder(self.chaikin, self.chaikin_sma)

        # Risk Management Calculations
        equity = self.equity
        risk_amount = equity * self.risk_percent / 100

        # 🌙✨ Long Entry
        if not self.position.is_long and bb_squeeze and chaikin_above:
            swing_low = self.swing_low[-1]
            risk_per_share = self.data.Close[-1] - swing_low
            if risk_per_share <= 0:
                print("🌙⚠️ Risk per share <=0. Skip LONG entry")
                return
            position_size = int(round(risk_amount / risk_per_share