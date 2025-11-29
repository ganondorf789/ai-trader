Here's the complete backtesting implementation for the VolCompressConvergence strategy:

```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Strategy, Backtest
from backtesting.lib import crossover

# Custom indicator functions 🌙
def bb_upper(close, timeperiod=20, nbdevup=2, nbdevdn=2):
    upper, _, _ = talib.BBANDS(close, timeperiod=timeperiod, 
                              nbdevup=nbdevup, nbdevdn=nbdevdn)
    return upper

def bb_middle(close, timeperiod=20, nbdevup=2, nbdevdn=2):
    _, middle, _ = talib.BBANDS(close, timeperiod=timeperiod,
                               nbdevup=nbdevup, nbdevdn=nbdevdn)
    return middle

def bb_lower(close, timeperiod=20, nbdevup=2, nbdevdn=2):
    _, _, lower = talib.BBANDS(close, timeperiod=timeperiod,
                              nbdevup=nbdevup, nbdevdn=nbdevdn)
    return lower

def calculate_bb_width(close, timeperiod=20, nbdevup=2, nbdevdn=2):
    upper, middle, lower = talib.BBANDS(close, timeperiod=timeperiod,
                                       nbdevup=nbdevup, nbdevdn=nbdevdn)
    return (upper - lower) / middle

def rolling_percentile(series, window, percentile):
    return series.rolling(window).quantile(percentile)

class VolCompressConvergence(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌙
    max_trades = 3   # Maximum concurrent trades
    
    def init(self):
        # Calculate core indicators using TA-Lib ✨
        self.bb_upper = self.I(bb_upper, self.data.Close, 20, 2, 2)
        self.bb_middle = self.I(bb_middle, self.data.Close, 20, 2, 2)
        self.bb_lower = self.I(bb_lower, self.data.Close, 20, 2, 2)
        
        # Volatility contraction detection 🌑
        self.bb_width = self.I(calculate_bb_width, self.data.Close, 20, 2, 2)
        self.bb_low_percentile = self.I(rolling_percentile, self.bb_width, 100, 0.1)
        
        # Price-VWAP convergence ✨
        self.vwap = self.I(ta.vwap, high=self.data.High, low=self.data.Low,
                          close=self.data.Close, volume=self.data.Volume, anchor='D')
        
        # ATR for risk management 🚀
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low,
                         self.data.Close, 14)

    def next(self):
        # Moon Dev position management 🌙✨
        if len(self.trades) >= self.max_trades:
            return
            
        current_close = self.data.Close[-1]
        current_vwap = self.vwap[-1]
        current_atr = self.atr[-1]
        
        # Entry conditions check 🌙
        volatility_cond = self.bb_width[-1] < self.bb_low_percentile[-1]
        convergence_cond = abs(current_close - current_vwap) <= current_atr
        
        if volatility_cond and convergence_cond:
            # Moon-sized risk calculation 🌚➗
            risk_amount = self.equity * self.risk_pct
            if current_atr > 0:
                position_size = risk_amount / (2 * current_atr)
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.buy(
                        size=position_size,
                        tag={
                            'entry_price': current_close,
                            'atr_entry': current_atr
                        }
                    )
                    print(f"🌙✨ MOON DEV ENTRY 🚀 | Price: {current_close:.2f} | Size: {position_size} | ATR: {current_atr:.2f}")

        # Check exits for all open trades 🌑
        for trade in self.trades:
            entry_price = trade.tag['entry_price']
            atr_entry = trade.tag['atr_entry']
            current_price = self.data.Close[-1]
            
            upper_target = entry_price + 2 * atr_entry
            lower_stop = entry_price - 2 * atr_entry
            
            if current_price >= upper_target or current_price <= lower_st