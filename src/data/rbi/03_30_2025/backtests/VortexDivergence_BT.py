Here's the complete implementation of the VortexDivergence strategy for backtesting.py:

```python
import pandas as pd
import pandas_ta as ta
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class VortexDivergence(Strategy):
    vi_period = 14
    kc_ema_period = 20
    kc_atr_period = 10
    kc_multiplier = 2.0
    risk_pct = 0.02
    atr_sl_multiplier = 1.5
    divergence_lookback = 5

    def init(self):
        # Calculate indicators using pandas_ta
        # Vortex Indicator
        vi = self.data.df['High'].ta.vortex(
            low=self.data.df['Low'],
            close=self.data.df['Close'],
            length=self.vi_period
        )
        self.vi_plus = self.I(lambda: vi[f'VMP_{self.vi_period}'], name='VI+')
        self.vi_minus = self.I(lambda: vi[f'VMM_{self.vi_period}'], name='VI-')

        # OBV using TA-Lib
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume)

        # Keltner Channel
        kc = self.data.df['High'].ta.kc(
            close=self.data.df['Close'],
            low=self.data.df['Low'],
            length=self.kc_ema_period,
            scalar=self.kc_multiplier,
            mamode='ema',
            tr=True
        )
        self.kc_upper = self.I(lambda: kc[f'KCUe_{self.kc_ema_period}_{self.kc_multiplier}'], name='KC Upper')
        self.kc_middle = self.I(lambda: kc[f'KCLe_{self.kc_ema_period}_{self.kc_multiplier}'], name='KC Middle')
        self.kc_lower = self.I(lambda: kc[f'KCL_{self.kc_ema_period}_{self.kc_multiplier}'], name='KC Lower')

        # ATR for risk management
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, 
                         self.data.Close, timeperiod=self.kc_atr_period)

    def next(self):
        if len(self.data) < max(self.vi_period, self.kc_ema_period, self.kc_atr_period) + 1:
            return

        current_price = self.data.Close[-1]
        atr_value = self.atr[-1]

        # Moon-themed risk management 🌙
        if not self.position:
            # Long entry logic
            if (crossover(self.vi_plus, self.vi_minus) and 
                self.bullish_divergence()):
                
                risk_amount = self.risk_pct * self.equity
                risk_per_share = self.atr_sl_multiplier * atr_value
                size = int(round(risk_amount / risk_per_share))
                
                if size > 0:
                    self.buy(
                        size=size,
                        sl=lambda trade: trade.entry_price - self.atr_sl_multiplier * atr_value,
                        tag="Moon Rocket Launch 🚀"
                    )
                    print(f"🌙🚀 LONG ENTRY | Price: {current_price:.2f} | Size: {size} | ATR SL: {atr_value:.2f}")

            # Short entry logic    
            elif (crossover(self.vi_minus, self.vi_plus) and 
                  self.bearish_divergence()):
                
                risk_amount = self.risk_pct * self.equity
                risk_per_share = self.atr_sl_multiplier * atr_value
                size = int(round(risk_amount / risk_per_share))
                
                if size > 0:
                    self.sell(
                        size=size,
                        sl=lambda trade: trade.entry_price + self.atr_sl_multiplier * atr_value,
                        tag="Dark Moon Descent 🌑"
                    )
                    print(f"🌙🌑 SHORT ENTRY | Price: {current_price:.2f} | Size: {size} | ATR SL: {atr_value:.2f}")

        # Exit logic
        if self.position.is_long and current_price < self.kc_lower[-1]:
            self.position.close()
            print(f"🌙✨ LONG EXIT | Price: {current_price:.2f} | KC Lower: {self.kc_lower[-