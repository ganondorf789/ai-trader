Here's the fixed code with all backtesting.lib references removed and Moon Dev themed improvements:

```python
import pandas as pd
import pandas_ta as ta
import talib
from backtesting import Backtest, Strategy
import numpy as np

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

    def bullish_divergence(self):
        """Moon-powered bullish divergence detection 🌙"""
        lookback = min(self.divergence_lookback, len(self.data.Close)-1)
        price_lows = self.data.Close[-lookback:]
        obv_lows = self.obv[-lookback:]
        
        # Check for lower price lows but higher OBV lows
        lower_price = np.all(price_lows[-1] < price_lows[:-1])
        higher_obv = np.all(obv_lows[-1] > obv_lows[:-1])
        
        if lower_price and higher_obv:
            print(f"🌙✨ BULLISH DIVERGENCE DETECTED | Price: {price_lows[-1]:.2f} | OBV: {obv_lows[-1]:.2f}")
            return True
        return False

    def bearish_divergence(self):
        """Dark moon bearish divergence detection 🌑"""
        lookback = min(self.divergence_lookback, len(self.data.Close)-1)
        price_highs = self.data.Close[-lookback:]
        obv_highs = self.obv[-lookback:]
        
        # Check for higher price highs but lower OBV highs
        higher_price = np.all(price_highs[-1] > price_highs[:-1])
        lower_obv = np.all(obv_highs[-1] < obv_highs[:-1])
        
        if higher_price and lower_obv:
            print(f"🌙🌑 BEARISH DIVERGENCE DETECTED | Price: {price_highs[-1]:.2f} | OBV: {obv_highs[-1]:.2f}")
            return True
        return False

    def next(self):
        if len(self.data) < max(self.vi_period, self.kc_ema_period, self.kc_atr_period) + 1:
            return

        current_price = self.data.Close[-1]
        atr_value = self.atr[-1]

        # Moon-themed risk management 🌙
        if not self.position:
            # Long entry logic - replaced crossover with manual check
            if ((self.vi_plus[-2] < self.vi_minus[-2] and self