Here's the complete backtesting implementation for VoltaicSkew strategy:

```python
# 🌙 Moon Dev's VoltaicSkew Backtest Implementation 🚀

# 1. Necessary imports
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# 2. Strategy class with indicators
class VoltaicSkew(Strategy):
    def init(self):
        # 🌌 Moon Dev Data Preparation
        self.data.columns = self.data.columns.str.strip().str.lower()
        self.data.drop(columns=[col for col in self.data.columns if 'unnamed' in col], 
                     errors='ignore', inplace=True)

        # 3. Indicator Calculations using TA-Lib ✨
        # 10-Day IV (Historical Volatility Proxy)
        self.iv = self.I(talib.STDDEV, self.data.close, timeperiod=960, name='IV')  # 960 periods = 10 days (15m)
        
        # IV Percentile (Rolling High/Low Method)
        self.iv_high = self.I(talib.MAX, self.iv, 2400, name='IV_High')  # 25-day window
        self.iv_low = self.I(talib.MIN, self.iv, 2400, name='IV_Low')
        self.iv_percentile = self.I(lambda iv, high, low: ((iv - low) / (high - low + 1e-6)) * 100,
                                  self.iv, self.iv_high, self.iv_low, name='IV_Pct')
        
        # Delta Skew (Price Skewness Proxy)
        self.delta_skew = self.I(ta.skew, self.data.close, length=672, name='Delta_Skew')  # 1-week window
        
        # Trend Slopes 🌗
        self.iv_slope = self.I(talib.LINEARREG_SLOPE, self.iv, 96, name='IV_Slope')  # 24hr trend
        self.skew_slope = self.I(talib.LINEARREG_SLOPE, self.delta_skew, 96, name='Skew_Slope')
        
        # IV/Spot Correlation
        self.iv_spot_corr = self.I(talib.CORREL, self.iv, self.data.close, 480, name='IV_Spot_Corr')  # 5-day
        self.corr_slope = self.I(talib.LINEARREG_SLOPE, self.iv_spot_corr, 48, name='Corr_Slope')

        print("🌑 VoltaicSkew Initialized with Moon Dev Indicators! 🌙✨")

    def next(self):
        # 🌔 Current Indicator Values
        iv_pct = self.iv_percentile[-1]
        skew_dir = self.skew_slope[-1]
        iv_dir = self.iv_slope[-1]
        
        print(f"🌙 IV%: {iv_pct:.1f}% | Skew Slope: {skew_dir:.2f} | IV Slope: {iv_dir:.2f}")

        # 4. Entry/Exit Logic with Moon Dev Flair 🚀
        if not self.position:
            # Entry Condition: High IV + Bearish Skew Divergence
            if iv_pct > 80 and skew_dir < 0 and iv_dir > 0:
                # 5. Risk Management Calculation
                risk_capital = self.equity * 0.01  # 1% risk
                entry_price = self.data.close[-1]
                position_size = risk_capital / entry_price
                
                # 🌙 Critical Position Sizing Fix
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.sell(size=position_size, 
                            reason=f"🌑 VOLTAIC ENTRY: Short at {entry_price:.2f}")
                    print(f"🚀 MOON SHORT! Size: {position_size} | Entry: {entry_price:.2f}")
                    
                    # Store trade parameters
                    self.entry_price = entry_price
                    self.risk_amount = risk_capital
        else:
            # 6. Exit Conditions 🌧️🎯
            current_pct = self.iv_percentile[-1]
            current_corr = self.corr_slope[-1]
            
            # Stop-Loss: IV Overheating
            if current_pct > 90:
                self.buy(size=self.position.size,
                       reason="🌧️ IV STORM