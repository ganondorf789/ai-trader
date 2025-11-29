Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np

class VortexReversion(Strategy):
    def init(self):
        # Clean data columns
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col.lower()])
        
        # Calculate indicators
        self.vi_plus = self.I(self._calculate_vortex, self.data.High, self.data.Low, self.data.Close, 14, name='VI_plus', which='plus')
        self.vi_minus = self.I(self._calculate_vortex, self.data.High, self.data.Low, self.data.Close, 14, name='VI_minus', which='minus')
        self.cmo = self.I(talib.CMO, self.data.Close, timeperiod=20, name='CMO')
        self.upper_bb = self.I(self._calc_upper_bb, self.data.Close, name='UpperBB')
        self.lower_bb = self.I(self._calc_lower_bb, self.data.Close, name='LowerBB')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR')
        self.volume_slope = self.I(talib.LINEARREG_SLOPE, self.data.Volume, timeperiod=5, name='VolumeSlope')
        
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='SwingHigh')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='SwingLow')
        
        self.trade_active = False
        print("🌙 VortexReversion Strategy Initialized with Moon Power! 🌙")
        print("✨ All indicators powered by TA-Lib and pandas-ta ✨")

    def _calculate_vortex(self, high, low, close, period, which):
        df = pd.DataFrame({'high': high, 'low': low, 'close': close})
        vortex = df.ta.vortex(length=period)
        return vortex[f'VIp_{period}'] if which == 'plus' else vortex[f'VIm_{period}']

    def _calc_upper_bb(self, close):
        upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        return upper

    def _calc_lower_bb(self, close):
        _, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        return lower

    def next(self):
        if len(self.data) < 20:
            return

        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        current_open = self.data.Open[-1]

        # Moon Dev Debug Prints
        print(f"🌙 Current Price: {current_close} | VI+={self.vi_plus[-1]:.2f} VI-={self.vi_minus[-1]:.2f} | CMO={self.cmo[-1]:.2f}")
        print(f"🌕 Swing High: {self.swing_high[-1]:.2f} | Swing Low: {self.swing_low[-1]:.2f} | ATR: {self.atr[-1]:.2f}")

        if not self.position:
            # Entry Logic
            risk_percent = 0.01  # 1% risk per trade
            
            # Short Entry Conditions
            if (self.vi_plus[-1] > self.vi_minus[-1] and
                self.data.High[-1] > self.data.High[-2] and
                self.cmo[-1] < self.cmo[-2] and
                self.volume_slope[-1] < 0 and
                current_close < current_open):
                
                swing_high = self.swing_high[-1]
                atr_value = self.atr[-1]
                stop_loss = swing_high + atr_value
                risk_amount = self.equity * risk_percent
                position_size = int(round(risk_amount / (stop_loss - current_close)))
                
                if position_size > 0:
                    self.sell(size=position_size)
                    self.stop_loss = stop_loss