I'll fix the code by removing the `backtesting.lib` import and ensuring all indicators are properly implemented. Here's the corrected version with Moon Dev themed debug prints:

```python
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
import talib

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolatilitySqueezeBreakout(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Bollinger Bands (20,2)
        self.upper_bb = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0], self.data.Close)
        self.lower_bb = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2], self.data.Close)
        
        # Keltner Channel (20,1.5)
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=20)
        self.atr_kc = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=20)
        self.kc_upper = self.I(lambda ema, atr: ema + 1.5*atr, self.ema, self.atr_kc)
        self.kc_lower = self.I(lambda ema, atr: ema - 1.5*atr, self.ema, self.atr_kc)
        
        # ADX (14)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        # Volume MA (20)
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
        # ATR for trailing stop (14)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        self.trade_counter = 0
        self.entry_bar = None

    def next(self):
        if len(self.data) < 20:
            return

        # Current indicator values
        squeeze = (self.upper_bb[-1] < self.kc_upper[-1]) and (self.lower_bb[-1] > self.kc_lower[-1])
        volume_condition = self.data.Volume[-1] > 1.5 * self.volume_ma[-1]
        adx_rising = self.adx[-1] > 25 and self.adx[-1] > self.adx[-2]

        # Entry logic
        if not self.position:
            if squeeze:
                # Long entry
                if self.data.Close[-1] > self.upper_bb[-1] and volume_condition and adx_rising:
                    stop_loss = self.lower_bb[-1]
                    risk_amount = self.risk_percent * self.equity
                    stop_distance = self.data.Close[-1] - stop_loss
                    
                    if stop_distance > 0:
                        position_size = int(round(risk_amount / stop_distance))
                        if position_size > 0:
                            self.buy(size=position_size)
                            self.entry_bar = len(self.data)
                            print(f"🚀🌙 Moon Dev LONG Signal! Entry: {self.data.Close[-1]:.2f}, Size: {position_size} units")
                            print(f"🌕 Starlight Trails: SL @ {stop_loss:.2f}, Risk: {risk_amount:.2f}")
                
                # Short entry
                elif self.data.Close[-1] < self.lower_bb[-1] and volume_condition and adx_rising:
                    stop_loss = self.upper_bb[-1]
                    risk_amount = self.risk_percent * self.equity
                    stop_distance = stop_loss - self.data.Close[-1]
                    
                    if stop_distance > 0: