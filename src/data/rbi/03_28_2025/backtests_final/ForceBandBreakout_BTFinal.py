I'll fix the incomplete code while maintaining all the strategy logic. Here's the complete corrected version with Moon Dev themed improvements:

```python
# 🌙 Moon Dev's ForceBand Breakout Backtest 🌙
import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Data preparation moon magic ✨
def prepare_data(path):
    data = pd.read_csv(path)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Remove unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    # Proper column mapping
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
    }, inplace=True)
    # Set datetime index
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

class ForceBandBreakout(Strategy):
    risk_per_trade = 0.02  # 2% risk per trade 🌙
    atr_multiplier = 1.5  # BB width exit threshold 🚀
    
    def init(self):
        # 🌌 Cosmic Indicators 🌌
        # Elder Force Index (13-period)
        self.force_index = self.I(talib.FORCEINDEX, 
                                 self.data.Close, 
                                 self.data.Volume, 
                                 timeperiod=13)
        
        # Bollinger Bands (20,2)
        self.bb_upper = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, 
                                                     nbdevdn=2, matype=0)[0], 
                              self.data.Close)
        self.bb_middle = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, 
                                                      nbdevdn=2, matype=0)[1], 
                               self.data.Close)
        self.bb_lower = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, 
                                                     nbdevdn=2, matype=0)[2], 
                              self.data.Close)
        
        # BB Width Indicator
        self.bb_width = self.I(lambda u, m, l: (u - l)/m, 
                              self.bb_upper, self.bb_middle, self.bb_lower)
        self.bb_width_avg = self.I(talib.SMA, self.bb_width, timeperiod=20)
        
        # ATR (14-period)
        self.atr = self.I(talib.ATR, 
                         self.data.High, 
                         self.data.Low, 
                         self.data.Close, 
                         timeperiod=14)
        
        # 🌑 Trailing stop variables
        self.trailing_stop_long = None
        self.trailing_stop_short = None

    def next(self):
        # 🛑 Risk Management First! Max 5 positions
        if len(self.trades) >= 5:
            return
        
        # 🌠 Entry Conditions
        current_close = self.data.Close[-1]
        current_atr = self.atr[-1]
        
        # Check BB contraction in last 3 bars
        contraction = any(self.bb_width[-i] < 0.5*self.bb_width_avg[-i] 
                         for i in range(1,4) if len(self.bb_width) >= i)
        
        # Long Entry Constellation 🌟
        if not self.position.is_long:
            # Moon Dev's Crossover Check 🌙✨
            force_cross = any((self.force_index[-i-1] < 0 and self.force_index[-i] > 0 
                             for i in range(1,4) if len(self.force_index) > i)
            price_break = any(current_close > self.bb_upper[-i] 
                            for i in range(1,4) if len(self.bb_upper) >= i)
            
            if contraction and force_cross and price_break:
                # 🌕 Calculate Moon Position Size 🌕
                stop_loss = current_close - current_atr
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = current_close - stop_loss
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    self.trailing_stop_long = current_close - 2*current_atr
                    print(f"🌕 NEW MOON LONG 🌕 |