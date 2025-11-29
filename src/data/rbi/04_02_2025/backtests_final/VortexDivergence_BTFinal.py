Here's the complete fixed code with Moon Dev themed debug prints and proper position sizing:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class VortexDivergence(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    initial_stop_pct = 0.02  # 2% initial stop
    adx_period = 14
    vi_period = 14
    mfi_period = 14

    def init(self):
        # Calculate indicators with proper TA-Lib integration
        self.vi_plus = self.I(lambda: talib.VORTEX(self.data.High, self.data.Low, self.vi_period)[0], name='VI+')
        self.vi_minus = self.I(lambda: talib.VORTEX(self.data.High, self.data.Low, self.vi_period)[1], name='VI-')
        self.mfi = self.I(talib.MFI, self.data.High, self.data.Low, self.data.Close, self.data.Volume, self.mfi_period, name='MFI')
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, self.adx_period, name='ADX')

        # Trailing stop variables
        self.trailing_high = None
        self.trailing_low = None
        self.trailing_stop = None

    def next(self):
        # Skip early bars without sufficient data
        if len(self.data) < 20 or len(self.adx) < 5:
            return

        # Entry signal detection
        current_close = self.data.Close[-1]
        bullish_crossover = (self.vi_plus[-2] < self.vi_minus[-2]) and (self.vi_plus[-1] > self.vi_minus[-1])
        bearish_crossover = (self.vi_minus[-2] < self.vi_plus[-2]) and (self.vi_minus[-1] > self.vi_plus[-1])

        # MFI divergence checks
        price_low_current, price_low_prev = self.data.Low[-1], self.data.Low[-2]
        mfi_current, mfi_prev = self.mfi[-1], self.mfi[-2]
        bullish_divergence = (price_low_current < price_low_prev) and (mfi_current > mfi_prev)
        bearish_divergence = (self.data.High[-1] > self.data.High[-2]) and (mfi_current < mfi_prev)

        # Entry logic
        if not self.position:
            # Long entry
            if bullish_crossover and bullish_divergence:
                risk_amount = self.equity * self.risk_per_trade
                stop_price = current_close * (1 - self.initial_stop_pct)
                risk_per_share = current_close - stop_price
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_price)
                    self.trailing_high = self.data.High[-1]
                    print(f"🚀 MOON DEV LONG ENTRY: {position_size} units @ {current_close:.2f} | Cosmic alignment detected! 🌙")
                    print(f"✨ Stellar Indicators: VI+ {self.vi_plus[-1]:.2f} > VI- {self.vi_minus[-1]:.2f} | MFI Bullish Divergence Confirmed")

            # Short entry
            elif bearish_crossover and bearish_divergence:
                risk_amount = self.equity * self.risk_per_trade
                stop_price = current_close * (1 + self.initial_stop_pct)
                risk_per_share = stop_price - current_close
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    self.sell(size=position_size, sl=stop_price)
                    self.trailing_low = self.data.Low[-1]
                    print(f"🌑 MOON DEV SHORT ENTRY: {position_size} units @ {current_close:.2f} | Stellar bearish convergence! 🌌")
                    print(f"💫 Dark Energy Indicators: VI- {self.vi_minus[-1]:.2f} > VI+ {self.vi_plus[-1]:.2f} | MFI Bearish Divergence Confirmed")

        # Position management
        else:
            # Calculate ADX dynamics
            adx_roc = self.adx[-1