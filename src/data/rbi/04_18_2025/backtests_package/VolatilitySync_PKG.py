Here's the fixed code with all backtesting.lib imports removed and proper Moon Dev themed replacements:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION FOR VOLATILITYSYNC STRATEGY
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np

class VolatilitySync(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    atr_multiplier = 1.5  # ATR-based stop loss
    
    def init(self):
        # 🌙 CORE INDICATORS
        # Elder's Force Index (13-period)
        self.efi = self.I(ta.efi, self.data.Close, self.data.Volume, length=13, name='EFI')
        
        # EFI volatility threshold (20-period std)
        self.efi_std = self.I(talib.STDDEV, self.efi, timeperiod=20, name='EFI_STD')
        
        # Bollinger Bands components
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(talib.BBANDS, self.data.Close, 
                                                              timeperiod=20, nbdevup=2, nbdevdn=2, 
                                                              matype=0, name=['BB_UP','BB_MID','BB_LOW'])
        self.bb_width = self.I(lambda u, l, m: (u - l)/m, self.bb_upper, self.bb_lower, 
                             self.bb_middle, name='BB_WIDTH')
        self.bb_width_avg = self.I(talib.SMA, self.bb_width, timeperiod=20, name='BB_WIDTH_AVG')
        
        # Trend filter
        self.sma200 = self.I(talib.SMA, self.data.Close, timeperiod=200, name='SMA200')
        
        # Keltner Channels (20-period EMA + 1.5xATR)
        self.ema20 = self.I(talib.EMA, self.data.Close, timeperiod=20, name='EMA20')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                        timeperiod=20, name='ATR20')
        self.keltner_upper = self.I(lambda e, a: e + 1.5*a, self.ema20, self.atr, name='KELT_UP')
        self.keltner_lower = self.I(lambda e, a: e - 1.5*a, self.ema20, self.atr, name='KELT_LO')
        
        # Volume surge detector
        self.vol_pct = self.I(ta.percentile, self.data.Volume, length=50, name='VOL_PCT')

    def next(self):
        current_close = self.data.Close[-1]
        moon_equity = self.equity  # 🌙 Total available capital
        
        # 🚀 LONG ENTRY CONDITIONS
        if (not self.position and
            (self.efi[-2] < self.efi_std[-2] and self.efi[-1] > self.efi_std[-1]) and  # Bullish crossover replacement
            self.bb_width[-1] < self.bb_width_avg[-1] and
            current_close > self.sma200[-1]):
            
            # 🌙 RISK MANAGEMENT CALCULATIONS
            atr_value = self.atr[-1]
            stop_price = current_close - (atr_value * self.atr_multiplier)
            risk_amount = moon_equity * self.risk_per_trade
            risk_per_unit = current_close - stop_price
            
            if risk_per_unit > 0:
                position_size = int(round(risk_amount / risk_per_unit))
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_price)
                    print(f"🌕🚀 MOON BULL SIGNAL! Size: {position_size} | Entry: {current_close:.2f} | SL: {stop_price:.2f}")

        # 🌑 SHORT ENTRY CONDITIONS
        elif (not self.position and
              (self.efi[-2] > -self.efi_std[-2] and self.efi[-1] < -self.efi_std[-1]) and  # Bearish crossunder replacement
              self.bb_width[-1] < self.bb_width_avg[-1] and
              current_close < self.sma200[-1]):
            
            atr_value = self.atr[-1]
            stop