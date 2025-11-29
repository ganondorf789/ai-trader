```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

class VolCompressBreakout(Strategy):
    def init(self):
        self.initial_equity = 1_000_000
        
        # Calculate indicators using self.I()
        # Bollinger Bands (20,2)
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0]
        self.bb_mid = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1]
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2]
        self.bb_width = self.I(lambda u, l, m: (u - l)/m, self.bb_upper, self.bb_lower, self.bb_mid)
        self.bb_width_ma = self.I(talib.SMA, self.bb_width, timeperiod=10)
        self.bb_width_ma_5 = self.I(talib.SMA, self.bb_width_ma, timeperiod=5)
        
        # Chaikin Volatility
        hl_diff = self.I(lambda h, l: h - l, self.data.High, self.data.Low)
        ema10_hl = self.I(talib.EMA, hl_diff, timeperiod=10)
        ema10_hl_shifted = self.I(lambda x: x.shift(10), ema10_hl)
        self.chaikin_vol = self.I(lambda x, y: ((x - y)/y)*100, ema10_hl, ema10_hl_shifted)
        self.chaikin_ma = self.I(talib.SMA, self.chaikin_vol, timeperiod=10)
        
        # Keltner Channel
        self.ema20 = self.I(talib.EMA, self.data.Close, timeperiod=20)
        self.atr20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=20)
        self.keltner_upper = self.I(lambda ema, atr: ema + 2*atr, self.ema20, self.atr20)
        self.keltner_lower = self.I(lambda ema, atr: ema - 2*atr, self.ema20, self.atr20)
        
        # Volume
        self.volume_sma20 = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
        # ATR for stops
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
    def next(self):
        if self.position:
            return
            
        # Setup conditions
        bb_narrowing = self.bb_width_ma[-1] < self.bb_width_ma_5[-1]
        chaikin_suppressed = self.chaikin_vol[-1] < self.chaikin_ma[-1]
        
        if bb_narrowing and chaikin_suppressed:
            # Volume surge check
            volume_surge = self.data.Volume[-1] >= 1.5 * self.volume_sma20[-1]
            
            if volume_surge:
                # Long entry
                if self.data.Close[-1] > self.keltner_upper[-1]:
                    entry_price = self.data.Close[-1]
                    stop_loss = min(self.keltner_lower[-1], entry_price - self.atr[-1])
                    risk_amount = 0.01 * self.initial_equity
                    risk_per_unit = entry_price - stop_loss
                    
                    if risk_per_unit <= 0:
                        return
                    
                    position_size = int(round(risk_amount / risk_per_unit))
                    if position_size > 0:
                        self.buy(size=position_size, sl=stop_loss)
                        self.position.data['max_high'] = self.data.High[-1]
                        self.position.data['entry_price'] = entry_price
                        self.position.data['bb_width'] = (self.bb_upper[-1] - self.bb_lower[-1])/self.bb_mid[-1]
                        print(f"🌕🚀 MOON DEV LONG ENTRY! Price: {entry_price}, Size: {position_size}")