Here's the fixed code with all backtesting.lib references removed and Moon Dev themed debug prints added:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class MomentumDivergence(Strategy):
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        self.data.df.index = pd.to_datetime(self.data.df.index)

        # Calculate indicators
        # Stochastic Oscillator (14,3,3)
        stoch_k, stoch_d = talib.STOCH(
            self.data.High, self.data.Low, self.data.Close,
            fastk_period=14, slowk_period=3, slowk_matype=0,
            slowd_period=3, slowd_matype=0
        )
        self.stoch_k = self.I(lambda: stoch_k, name='Stoch %K')
        self.stoch_d = self.I(lambda: stoch_d, name='Stoch %D')
        
        # OBV and trend levels
        obv = talib.OBV(self.data.Close, self.data.Volume)
        self.obv = self.I(lambda: obv, name='OBV')
        self.obv_low = self.I(lambda: talib.MIN(obv, 20), name='OBV Low')
        self.obv_high = self.I(lambda: talib.MAX(obv, 20), name='OBV High')
        
        # Price swing levels
        self.price_low = self.I(lambda: talib.MIN(self.data.Low, 20), name='Price Low')
        self.price_high = self.I(lambda: talib.MAX(self.data.High, 20), name='Price High')
        
        # Volume filter
        self.volume_ma = self.I(lambda: talib.SMA(self.data.Volume, 20), name='Volume MA')

    def next(self):
        # Skip trades during low volume periods
        if self.data.Volume[-1] < self.volume_ma[-1] * 0.8:
            print("🌙 Low volume detected - Standing by for better conditions...")
            return

        # Get indicator values
        stoch_k = self.stoch_k[-1]
        prev_stoch_k = self.stoch_k[-2]
        price_low = self.price_low[-1]
        prev_price_low = self.price_low[-2]
        obv_low = self.obv_low[-1]
        prev_obv_low = self.obv_low[-2]
        price_high = self.price_high[-1]
        prev_price_high = self.price_high[-2]
        obv_high = self.obv_high[-1]
        prev_obv_high = self.obv_high[-2]

        # Detect divergences
        bullish_divergence = (price_low < prev_price_low) and (obv_low > prev_obv_low)
        bearish_divergence = (price_high > prev_price_high) and (obv_high < prev_obv_high)

        # Entry logic
        if not self.position:
            # Long entry
            if (prev_stoch_k <= 20 and stoch_k > 20) and bullish_divergence:
                risk_percent = 0.01
                entry_price = self.data.Close[-1]
                stop_loss = price_low
                risk_per_share = entry_price - stop_loss
                
                if risk_per_share <= 0:
                    print("🌙 Invalid SL for LONG. Aborting launch!")
                    return
                
                position_size = int(round((self.equity * risk_percent) / risk_per_share))
                if position_size > 0:
                    self.buy(size=position_size, stop=stop_loss)
                    print(f"🚀🌙 MOONSHOT LONG! Size: {position_size}, Entry: {entry_price}, SL: {stop_loss}")
                else:
                    print("🌙 Position size too small for LONG - Need more fuel!")

            # Short entry
            elif (prev_stoch_k >= 80 and stoch_k < 80) and bearish_divergence:
                risk_percent = 0.01
                entry_price = self.data.Close[-1]
                stop_loss = price_high
                risk_per_share = stop_loss - entry_price
                
                if risk_per_share <= 0:
                    print("🌙 Invalid SL for SH