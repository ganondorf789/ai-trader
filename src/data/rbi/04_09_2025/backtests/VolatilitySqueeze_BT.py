```python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import talib
import pandas_ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Moon Dev Data Preparation 🌙
def prepare_data(path):
    data = pd.read_csv(path)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    
    # Proper column mapping
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    
    # Convert index to datetime
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

class VolatilitySqueeze(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # Bollinger Bands Components 🎯
        self.bb_middle = self.I(talib.SMA, self.data.Close, 20, name='BB_MIDDLE')
        self.bb_std = self.I(talib.STDDEV, self.data.Close, 20, name='BB_STD')
        self.bb_upper = self.I(lambda m, s: m + 2*s, self.bb_middle, self.bb_std, name='BB_UPPER')
        self.bb_lower = self.I(lambda m, s: m - 2*s, self.bb_middle, self.bb_std, name='BB_LOWER')
        
        # Bollinger Band Width Calculation 🌗
        self.bbw = self.I(lambda u, l, m: (u - l)/m, self.bb_upper, self.bb_lower, self.bb_middle, name='BBW')
        
        # RSI Indicator ✨
        self.rsi = self.I(talib.RSI, self.data.Close, 14, name='RSI')
        
        # BBW Percentile Calculations 🚀
        self.bbw_low = self.I(lambda x: pandas_ta.percentile(x, length=100, q=10), self.bbw, name='BBW_10th')
        self.bbw_high = self.I(lambda x: pandas_ta.percentile(x, length=100, q=50), self.bbw, name='BBW_50th')
        
    def next(self):
        # Moon Dev Debug Prints 🌙
        print(f"\n🌙 Current BBW: {self.bbw[-1]:.4f} | 10th: {self.bbw_low[-1]:.4f} | 50th: {self.bbw_high[-1]:.4f}")
        print(f"✨ RSI: {self.rsi[-1]:.2f} | Close: {self.data.Close[-1]:.2f} | BB Mid: {self.bb_middle[-1]:.2f}")
        
        # Entry Logic 🚀
        if not self.position:
            if (self.bbw[-1] <= self.bbw_low[-1] and
                self.rsi[-1] < 30 and
                self.data.Close[-1] > self.bb_middle[-1]):
                
                entry_price = self.data.Close[-1]
                stop_loss = self.bb_lower[-1]
                risk_per_share = entry_price - stop_loss
                
                if risk_per_share > 0:
                    risk_amount = self.equity * self.risk_per_trade
                    position_size = int(round(risk_amount / risk_per_share))
                    
                    if position_size > 0:
                        self.buy(size=position_size, sl=stop_loss, 
                                tag=f"🌙 Volatility Squeeze Entry! SL: {stop_loss:.2f}")
                        print(f"🚀 MOON DEV ALERT: LONG {position_size} @ {entry_price:.2f}")
        
        # Exit Logic 🌑
        else:
            if self.bbw[-1] > self.bbw_high[-1]:
                self.position.close()
                print(f"🌑 MOON DEV EXIT: Closing @ {self.data.Close[-1]:.2f}")

# Moon Dev Backtest Execution 🌙
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = prepare_data(data_path)

bt = Back