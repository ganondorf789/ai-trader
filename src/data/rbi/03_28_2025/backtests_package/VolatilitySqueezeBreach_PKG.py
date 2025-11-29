I'll fix the code by removing the `backtesting.lib` import and replacing any usage of it with proper alternatives. Here's the corrected version:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class VolatilitySqueezeBreach(Strategy):
    def init(self):
        # Moon Dev Indicator Initialization 🌙✨
        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(talib.BBANDS, self.data.Close, 
                                                             timeperiod=20, nbdevup=2, nbdevdn=2, 
                                                             matype=0, name=['BB_Upper', 'BB_Middle', 'BB_Lower'])
        
        # BB Width Calculation
        self.bb_width = self.I(lambda u, l: u - l, self.bb_upper, self.bb_lower, name='BB_Width')
        self.bb_width_sma = self.I(talib.SMA, self.bb_width, timeperiod=10, name='BB_Width_SMA_10')
        
        # Keltner Channels
        self.typical = self.I(lambda h, l, c: (h + l + c)/3, 
                            self.data.High, self.data.Low, self.data.Close, name='Typical_Price')
        self.ema_typical = self.I(talib.EMA, self.typical, timeperiod=20, name='EMA_Typical_20')
        self.atr_kc = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                           timeperiod=20, name='ATR_KC_20')
        self.upper_kc = self.I(lambda ema, atr: ema + 1.5*atr, 
                             self.ema_typical, self.atr_kc, name='Upper_KC')
        self.lower_kc = self.I(lambda ema, atr: ema - 1.5*atr, 
                             self.ema_typical, self.atr_kc, name='Lower_KC')
        
        # Trailing Stop Indicators
        self.atr_trailing = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                                 timeperiod=14, name='ATR_Trailing_14')
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='Volume_SMA_20')
        
        # Moon Dev Trailing Trackers �🌑
        self.trailing_high = None
        self.trailing_low = None

    def next(self):
        # Moon Dev Debug Console 🌙📊
        if len(self.data) % 100 == 0:
            print(f"\n🌙 Moon Dev Status Update [Bar {len(self.data)}]")
            print(f"   BB Width: {self.bb_width[-1]:.2f} < SMA10: {self.bb_width_sma[-1]:.2f}? {'✅' if self.bb_width[-1] < self.bb_width_sma[-1] else '❌'}")
            print(f"   Price: {self.data.Close[-1]:.2f} | Upper KC: {self.upper_kc[-1]:.2f} | Lower KC: {self.lower_kc[-1]:.2f}")
            print(f"   ATR14: {self.atr_trailing[-1]:.2f} | Volume: {self.data.Volume[-1]:.2f} (3x SMA20: {3*self.volume_sma[-1]:.2f})")

        # Entry Logic 🌟
        if not self.position:
            squeeze_active = self.bb_width[-1] < self.bb_width_sma[-1]
            
            if squeeze_active:
                # Long Entry 🚀
                if self.data.Close[-1] > self.upper_kc[-1]:
                    risk_amount = self.equity * 0.01
                    stop_loss = 1.5 * self.atr_trailing[-1]
                    position_size = int(round(risk_amount / stop_loss))
                    
                    if position_size > 0:
                        self.buy(size=position_size, sl=self.data.Close[-1] - stop_loss)
                        self.trailing_high = self.data.High[-1]
                        print(f"\n🚀🌙 MOON DEV LONG SIGNAL! 🚀")
                        print(f"   Size: {position_size} | Entry: {self.data.Close[-1]:.2f}")
                        print(f"   SL: {self.data.Close