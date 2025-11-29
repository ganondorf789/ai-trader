Here's the fixed code with all backtesting.lib imports removed and proper Moon Dev themed debug prints: 🌙✨

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from pandas_ta import vortex as vortex_ta

class VortexSqueeze(Strategy):
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators with self.I()
        self.vi_plus, self.vi_minus = self.I(self._calculate_vortex, self.data.High, self.data.Low, self.data.Close, 14)
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=10)
        self.upper_kc = self.I(lambda ema, atr: ema + 2*atr, self.ema, self.atr)
        self.lower_kc = self.I(lambda ema, atr: ema - 2*atr, self.ema, self.atr)
        self.sar = self.I(talib.SAR, self.data.High, self.data.Low, acceleration=0.02, maximum=0.2)
        
        print("🌙✨ VortexSqueeze Strategy Initialized with Moon Power! ✨🌙")

    def _calculate_vortex(self, high, low, close, length):
        """Calculate Vortex Indicator with pandas_ta"""
        vi_df = vortex_ta(high=high, low=low, close=close, length=length)
        return vi_df[f'VTXP_{length}'], vi_df[f'VTXM_{length}']

    def next(self):
        price = self.data.Close[-1]
        sar_value = self.sar[-1]
        
        # Moon Dev debug prints 🌙
        if len(self.data) % 100 == 0:
            print(f"🌙 Current Price: {price} | VI+:{self.vi_plus[-1]:.2f} VI-:{self.vi_minus[-1]:.2f}")
            print(f"   KC Upper: {self.upper_kc[-1]:.2f} Lower: {self.lower_kc[-1]:.2f} | SAR: {sar_value:.2f}")

        if not self.position:
            # Long Entry Logic 🚀
            if ((self.vi_plus[-2] < self.vi_minus[-2] and self.vi_plus[-1] > self.vi_minus[-1]) and
               (price > self.upper_kc[-1]) and
               (self.data.Volume[-1] > self.data.Volume[-2])):
                
                risk_amount = self.equity * 0.01
                stop_loss = self.lower_kc[-1]
                risk_per_share = price - stop_loss
                
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    print(f"🚀 BULLISH BREAKOUT! Buying {size} shares at {price}")
                    self.buy(size=size, sl=stop_loss)

            # Short Entry Logic 🐻
            elif ((self.vi_minus[-2] < self.vi_plus[-2] and self.vi_minus[-1] > self.vi_plus[-1]) and
                 (price < self.lower_kc[-1]) and
                 (self.data.Volume[-1] < self.data.Volume[-2])):
                
                risk_amount = self.equity * 0.01
                stop_loss = self.upper_kc[-1]
                risk_per_share = stop_loss - price
                
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    print(f"🐻 BEARISH BREAKDOWN! Shorting {size} shares at {price}")
                    self.sell(size=size, sl=stop_loss)
        else:
            # Exit Logic 💸
            if self.position.is_long:
                if ((self.vi_minus[-2] < self.vi_plus[-2] and self.vi_minus[-1] > self.vi_plus[-1]) or 
                    (price < sar_value) or 
                    (price < self.upper_kc[-1])):
                    print(f"🌙 CLOSING LONG at {price} (VI:{self.