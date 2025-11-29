I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete debugged version with Moon Dev improvements:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np
from backtesting import Strategy, Backtest

class VoltaicSurge(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Clean and prepare data with Moon Dev precision
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.dropna()
        
        # Calculate indicators using self.I() with proper talib/pandas_ta implementations
        # Bollinger Bands (20,2)
        upper, middle, lower = talib.BBANDS(self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        self.bb_upper = self.I(lambda: upper, name='BB_UPPER')
        self.bb_middle = self.I(lambda: middle, name='BB_MIDDLE')
        self.bb_lower = self.I(lambda: lower, name='BB_LOWER')
        
        # Bollinger Bandwidth calculations - Moon Dev optimized
        self.bb_bandwidth = self.I(lambda: (upper - lower)/middle, name='BB_BANDWIDTH')
        self.bb_bandwidth_30min = self.I(talib.MIN, self.bb_bandwidth, timeperiod=30, name='BB_BW_MIN30')
        
        # Volume SMA20 - Lunar Volume Analysis
        self.volume_sma20 = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VOL_SMA20')
        
        # Keltner Channels (20,1.5) - Cosmic Channel Analysis
        kc = ta.kc(self.data.High, self.data.Low, self.data.Close, length=20, scalar=1.5)
        self.kc_upper = self.I(lambda: kc['KCU_20_1.5'].values, name='KC_UPPER')
        self.kc_lower = self.I(lambda: kc['KCL_20_1.5'].values, name='KC_LOWER')
        
        # Swing low (20-period) - Moon Phase Detection
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='SWING_LOW')
        
        print("🌙 VoltaicSurge Strategy Initialized with Moon Dev Precision! 🚀")
        print("✨ Cosmic Indicators Aligned | No backtesting.lib Dependencies Found! 🌌")

    def next(self):
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        
        # Moon Dev Debug Prints
        if len(self.data) % 1000 == 0:
            print(f"🌙 Lunar Cycle Update: Bar {len(self.data)} | Price: {current_close:.2f} | Vol: {current_volume:.0f}")
            print(f"🌌 BB Width: {self.bb_bandwidth[-1]:.4f} | KC Range: {self.kc_upper[-1]-self.kc_lower[-1]:.2f}")
        
        if not self.position:
            # Long Entry Conditions - Cosmic Alignment Check
            cond1 = np.isclose(self.bb_bandwidth[-1], self.bb_bandwidth_30min[-1], rtol=1e-5)
            cond2 = current_volume > 2 * self.volume_sma20[-1]
            cond3 = current_close > self.kc_upper[-1]
            
            if cond1 and cond2 and cond3:
                # Lunar Risk Management Calculations
                equity = self.equity
                risk_amount = equity * self.risk_per_trade
                entry_price = current_close
                stop_loss_price = self.swing_low[-1]
                risk_per_share = entry_price - stop_loss_price
                
                if risk_per_share <= 0:
                    print("🌙⚡ Aborting Trade: Negative Risk Detected!")
                    print("💫 Cosmic Alignment Broken - Invalid Risk Parameters")
                    return
                
                # Moon Dev Position Sizing - whole shares only
                position_size = int(risk_amount / risk_per_share)
                
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_loss_price, tp=self.kc_lower[-1])
                    print(f"🚀 Cosmic Breakout Detected! Entering Long {position_size} shares")
                    print(f"🌕 Entry: {entry_price:.2f