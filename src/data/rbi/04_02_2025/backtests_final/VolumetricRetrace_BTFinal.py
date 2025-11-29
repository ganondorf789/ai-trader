Here's the debugged version of your VolumetricRetrace strategy with Moon Dev enhancements 🌙✨:

```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

class VolumetricRetrace(Strategy):
    # Strategy parameters
    chop_period = 14
    chop_threshold_low = 38
    chop_threshold_high = 62
    ma_period = 50
    atr_period = 14
    atr_ma_period = 20
    volume_ma_period = 20
    risk_pct = 0.01  # 1% risk per trade
    
    def init(self):
        # Moon Dev Indicator Suite 🌙
        self.chop = self.I(ta.choppiness, self.data.High, self.data.Low, self.data.Close, 
                          length=self.chop_period, scalar=100, name="CHOP")
        self.ma = self.I(talib.SMA, self.data.Close, timeperiod=self.ma_period, name="SMA")
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                         timeperiod=self.atr_period, name="ATR")
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=self.atr_ma_period, name="ATR_MA")
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_ma_period, 
                               name="VOL_MA")
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name="SWING_LOW")
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name="SWING_HIGH")
        
        print("🌙 VolumetricRetrace Strategy Initialized! ✨")

    def next(self):
        price = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        
        # Moon Dev Debug Dashboard 🌙
        print(f"\n🌙 DATE: {self.data.index[-1]} | PRICE: {price:.2f}")
        print(f"✨ CHOP: {self.chop[-1]:.2f} | ATR: {self.atr[-1]:.2f}")
        print(f"📈 VOL: {current_volume:.2f} vs MA: {self.volume_ma[-1]:.2f}")

        if not self.position:
            # Trend confirmation 🌙
            if self.chop[-1] < self.chop_threshold_low:
                # Long setup 🚀
                if (price > self.ma[-1] and 
                    self.data.Low[-1] <= self.ma[-1] and 
                    current_volume > self.volume_ma[-1]):
                    
                    stop_loss = self.swing_low[-1]
                    risk_amount = self.risk_pct * self.broker.equity
                    risk_per_unit = price - stop_loss
                    
                    if risk_per_unit > 0:
                        position_size = int(round(risk_amount / risk_per_unit))
                        if position_size > 0:
                            self.buy(size=position_size, sl=stop_loss, 
                                   tag="MoonLong🌙")
                            print(f"\n🚀🌙 BULLISH ENTRY!")
                            print(f"ENTRY: {price:.2f} | SIZE: {position_size}")
                            print(f"SL: {stop_loss:.2f} | RR: 1:2")

                # Short setup 🌧️
                elif (price < self.ma[-1] and 
                      self.data.High[-1] >= self.ma[-1] and 
                      current_volume > self.volume_ma[-1]):
                    
                    stop_loss = self.swing_high[-1]
                    risk_amount = self.risk_pct * self.broker.equity
                    risk_per_unit = stop_loss - price
                    
                    if risk_per_unit > 0:
                        position_size = int(round(risk_amount / risk_per_unit))
                        if position_size > 0:
                            self.sell(size=position_size, sl=stop_loss, 
                                    tag="MoonShort🌙")
                            print(f"\n🌧️🌙 BEARISH ENTRY!")
                            print(f"ENTRY: {price:.2f} | SIZE: {position_size}")
                            print(f"SL: {stop_loss:.2f} | RR: 1:2")
        else:
            # Position management 🌙
            if self.position.is_long:
                # Long exit conditions