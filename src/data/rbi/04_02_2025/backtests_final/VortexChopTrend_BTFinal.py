I'll fix the code while maintaining the original strategy logic. Here's the debugged version with Moon Dev enhancements:

```python
import pandas as pd
import pandas_ta as ta
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class VortexChopTrend(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # ====== INDICATOR CALCULATIONS ======
        # Vortex Indicator (14-period)
        vi_plus, vi_minus = ta.vortex(
            high=self.data.High,
            low=self.data.Low,
            close=self.data.Close,
            length=14
        )
        self.vi_plus = self.I(lambda: vi_plus, name='VI+ ✨')
        self.vi_minus = self.I(lambda: vi_minus, name='VI- ✨')

        # Choppiness Index (14-period)
        ci = ta.ci(
            high=self.data.High,
            low=self.data.Low,
            close=self.data.Close,
            length=14
        )
        self.ci = self.I(lambda: ci, name='ChopIdx 🌊')

        # ADX (14-period)
        self.adx = self.I(talib.ADX,
                          self.data.High,
                          self.data.Low,
                          self.data.Close,
                          timeperiod=14,
                          name='ADX 📈')

        # Swing Highs/Lows (20-period)
        self.swing_high = self.I(talib.MAX, 
                               self.data.High, 
                               timeperiod=20,
                               name='SwingHigh ⛰️')
        self.swing_low = self.I(talib.MIN,
                              self.data.Low,
                              timeperiod=20,
                              name='SwingLow 🏞️')

    def next(self):
        # ====== MOON DEV DEBUG PRINT ======
        if len(self.data) % 100 == 0:
            print(f"\n🌙 CURRENT STATE 🌙\nVI+={self.vi_plus[-1]:.2f} | VI-={self.vi_minus[-1]:.2f}")
            print(f"CI={self.ci[-1]:.1f} | ADX={self.adx[-1]:.1f} | Price={self.data.Close[-1]:.2f}\n")

        # ====== ENTRY LOGIC ======
        if not self.position:
            # Long Entry 🌙🚀
            if (crossover(self.vi_plus, self.vi_minus) and
               (self.ci[-1] < 38.2) and
               (self.adx[-1] > 20)):
                
                sl_price = self.swing_low[-1]
                risk_per_share = self.data.Close[-1] - sl_price
                
                if risk_per_share > 0:
                    position_size = int(round(
                        (self.equity * self.risk_pct) / risk_per_share
                    ))
                    self.buy(size=position_size, 
                            sl=sl_price,
                            tag="Moon Bullish Entry 🌕🚀")
                    print(f"\n🚀🌙 BULLISH SIGNAL! Entry: {self.data.Close[-1]:.2f}")
                    print(f"Size: {position_size} | SL: {sl_price:.2f} | Risk: {risk_per_share:.2f}")

            # Short Entry 🌙📉
            elif (crossover(self.vi_minus, self.vi_plus) and
                 (self.ci[-1] < 38.2) and
                 (self.adx[-1] > 20)):
                
                sl_price = self.swing_high[-1]
                risk_per_share = sl_price - self.data.Close[-1]
                
                if risk_per_share > 0:
                    position_size = int(round(
                        (self.equity * self.risk_pct) / risk_per_share
                    ))
                    self.sell(size=position_size,
                             sl=sl_price,
                             tag="Moon Bearish Entry 🌕📉")
                    print(f"\n📉🌙 BEARISH SIGNAL! Entry: {self.data.Close[-1]:.2f}")
                    print(f"Size: {position_size} | SL: {sl_price:.2f} | Risk: {risk_per_share:.2f}")

        # ====== EXIT LOGIC ======
        elif self.position:
            # Exit conditions for long position
            if self.position.is_long and (self.adx[-1] < 25 or crossover(self.vi_minus, self.vi_plus)):
                self.position