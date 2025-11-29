I'll fix the code by removing the `backtesting.lib` import and replacing any potential usage with proper alternatives. Here's the corrected version with Moon Dev themed improvements:

```python
# 🌙 Moon Dev's FundingPulse Backtest Script �✨

# 1. Necessary imports - CLEANED of backtesting.lib 🧹
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

# 2. Strategy implementation - PURE MOON POWER 🌕
class FundingPulse(Strategy):
    # Strategy parameters 🌌
    funding_threshold = 0.001  # 0.1% extreme funding rate
    adx_threshold_entry = 25   # ADX <25 for ranging market
    adx_threshold_exit = 35    # ADX >35 to exit
    median_window = 2880        # 30 days in 15m intervals (24*4*30=2880)
    risk_pct = 0.01            # 1% risk per trade
    stop_loss_pct = 0.02       # 2% fixed stop loss

    def init(self):
        # 🌗 Calculate indicators using TA-Lib and self.I() - NO backtesting.lib!
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ADX')
        self.median_funding = self.I(lambda x: x.rolling(self.median_window).median(), 
                                    self.data.funding_rate, name='MEDIAN_FUND')
        self.std_funding = self.I(lambda x: x.rolling(self.median_window).std(), 
                                 self.data.funding_rate, name='STD_FUND')
        
        print("🌙 FundingPulse Strategy Initialized with Pure Moon Energy! 🌕✨")

    def next(self):
        # 🌑 Skip if not enough data or missing values
        if len(self.data) < self.median_window or \
            np.isnan(self.median_funding[-1]) or \
            np.isnan(self.std_funding[-1]) or \
            np.isnan(self.adx[-1]):
            return

        # 🌓 Current market conditions
        price = self.data.Close[-1]
        current_funding = self.data.funding_rate[-1]
        median = self.median_funding[-1]
        std = self.std_funding[-1]
        current_adx = self.adx[-1]
        band_upper = median + std
        band_lower = median - std

        # 🌔 Moon-themed debug prints ✨
        print(f"🌙 Moon Pulse Check │ ADX: {current_adx:.1f} │ Funding: {current_funding*100:.3f}% │ Median Band: [{band_lower*100:.3f}% - {band_upper*100:.3f}%]")

        # 3. Exit logic first 🌗
        if self.position:
            if self.position.is_long and (current_adx > self.adx_threshold_exit or 
                                         (band_lower < current_funding < band_upper)):
                self.position.close()
                print(f"🌙 MOON DEV LONG EXIT 🌗 │ Reason: {'ADX Trend' if current_adx >35 else 'Funding Normalized'} │ Moon Phase Complete")

            elif self.position.is_short and (current_adx > self.adx_threshold_exit or 
                                            (band_lower < current_funding < band_upper)):
                self.position.close()
                print(f"🌙 MOON DEV SHORT EXIT 🌓 │ Reason: {'ADX Trend' if current_adx >35 else 'Funding Normalized'} │ Moon Cycle Ended")

        # 4. Entry logic with risk management 🌚
        if not self.position:
            # Long entry: extreme negative funding + low trend
            if current_funding < -self.funding_threshold and current_adx < self.adx_threshold_entry:
                sl_price = price * (1 - self.stop_loss_pct)
                risk_amount = self.equity * self.risk_pct
                risk_per_share = price - sl_price
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=sl_price)
                    print(f"🌙 MOON DEV LONG ENTRY 🌕 │ Size: {position_size} │ Stop: {sl_price:.2f} │ Moon Power Engaged!")

            # Short entry: extreme positive funding + low trend
            elif current_funding > self.funding_threshold and current_adx < self.adx_threshold_entry:
                sl_price = price * (1 + self.stop_loss