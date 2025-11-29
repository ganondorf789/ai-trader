I'll fix the backtesting.lib imports and ensure all indicators use proper packages. Here's the corrected code:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION - VOL DIVERGENCE STRATEGY 🚀

# -------------------
# 1. IMPORTS
# -------------------
from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta

# -------------------
# 2. DATA PREPARATION
# -------------------
def load_data(path):
    # Load and clean data
    data = pd.read_csv(path)
    
    # Clean column names 🌙
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    
    # Map required columns with proper case
    column_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    data.rename(columns=column_mapping, inplace=True)
    
    # Ensure datetime format
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    
    return data

# -------------------
# 3. STRATEGY CLASS
# -------------------
class VolDivergence(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    atr_period = 14
    bb_period = 20
    divergence_window = 5
    
    def init(self):
        # 🌙 BOLLINGER BANDS CALCULATION
        self.upper, self.middle, self.lower = self.I(talib.BBANDS, 
                                                    self.data.Close, 
                                                    timeperiod=self.bb_period,
                                                    nbdevup=2, nbdevdn=2,
                                                    matype=0)
        
        # 🌙 BOLLINGER BAND WIDTH (BBW)
        self.bbw = (self.upper - self.lower) / self.middle
        
        # 🌙 1-MONTH LOW (1920 periods for 15m data)
        self.bbw_low = self.I(talib.MIN, self.bbw, timeperiod=1920)
        
        # 🌙 VOLATILITY EXPANSION THRESHOLD (90th percentile)
        self.bbw_90th = self.I(lambda x: x.rolling(4320).quantile(0.9),  # 3-month lookback
                              self.bbw,
                              name='BBW_90th')
        
        # 🌙 PUT/CALL DIVERGENCE INDICATORS
        self.put_call_slope = self.I(talib.LINEARREG_SLOPE,
                                    self.data.df['put_call_ratio'],  # Assumes column exists
                                    timeperiod=self.divergence_window)
        
        self.price_slope = self.I(talib.LINEARREG_SLOPE,
                                 self.data.Close,
                                 timeperiod=self.divergence_window)
        
        # 🌙 VOLATILITY MEASURE (ATR for stops)
        self.atr = self.I(talib.ATR,
                         self.data.High,
                         self.data.Low,
                         self.data.Close,
                         timeperiod=self.atr_period)
        
        print("🌙✨ MOON DEV INDICATORS INITIALIZED! 🚀")

    def next(self):
        # Skip early bars where indicators aren't calculated
        if len(self.bbw) < 1920 or len(self.bbw_90th) < 1:
            return
        
        # 🌙 CURRENT VALUES
        price = self.data.Close[-1]
        bbw = self.bbw[-1]
        bbw_low = self.bbw_low[-1]
        bbw_90th = self.bbw_90th[-1]
        put_call_slope = self.put_call_slope[-1]
        price_slope = self.price_slope[-1]
        atr = self.atr[-1]
        
        # 🌙 DIVERGENCE CHECK
        bullish_div = (put_call_slope < 0) and (price_slope > 0)
        bearish_div = (put_call_slope > 0) and (price_slope < 0)
        
        # 🌙 ENTRY CONDITIONS
        if bbw < bbw_low and not self.position:
            # Calculate position size
            risk_amount = self.equity * self.risk_per_trade
            stop_distance = 2 * atr
            
            if bullish_div:
                # 🌙 LONG ENTRY
                units =