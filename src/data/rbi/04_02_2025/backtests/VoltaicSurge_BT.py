```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Strategy, Backtest

class VoltaicSurge(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators using self.I()
        # Bollinger Bands (20,2)
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, 20, 2, 2, 0, name='BB_UPPER', position=0)
        self.bb_middle = self.I(talib.BBANDS, self.data.Close, 20, 2, 2, 0, name='BB_MIDDLE', position=1)
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, 20, 2, 2, 0, name='BB_LOWER', position=2)
        
        # Bollinger Bandwidth calculations
        self.bb_bandwidth = self.I(lambda u, l, m: (u - l)/m, self.bb_upper, self.bb_lower, self.bb_middle, name='BB_BANDWIDTH')
        self.bb_bandwidth_30min = self.I(talib.MIN, self.bb_bandwidth, 30, name='BB_BW_MIN30')
        
        # Volume SMA20
        self.volume_sma20 = self.I(talib.SMA, self.data.Volume, 20, name='VOL_SMA20')
        
        # Keltner Channels (20,1.5)
        kc = ta.kc(self.data.High, self.data.Low, self.data.Close, length=20, scalar=1.5)
        self.kc_upper = self.I(lambda: kc['KCU_20_1.5'].values, name='KC_UPPER')
        self.kc_lower = self.I(lambda: kc['KCL_20_1.5'].values, name='KC_LOWER')
        
        # Swing low (20-period)
        self.swing_low = self.I(talib.MIN, self.data.Low, 20, name='SWING_LOW')
        
        print("🌙 VoltaicSurge Strategy Initialized with Moon Dev Precision! 🚀")

    def next(self):
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        
        # Moon Dev Debug Prints
        if len(self.data) % 1000 == 0:
            print(f"🌙 Lunar Cycle Update: Bar {len(self.data)} | Price: {current_close} | Vol: {current_volume}")
        
        if not self.position:
            # Long Entry Conditions
            cond1 = self.bb_bandwidth[-1] == self.bb_bandwidth_30min[-1]
            cond2 = current_volume > 2 * self.volume_sma20[-1]
            cond3 = current_close > self.kc_upper[-1]
            
            if cond1 and cond2 and cond3:
                # Risk Management Calculations
                equity = self._broker.portfolio_value
                risk_amount = equity * self.risk_per_trade
                entry_price = current_close
                stop_loss_price = self.swing_low[-1]
                
                if entry_price <= stop_loss_price:
                    print("🌙⚡ Aborting Trade: Negative Risk Detected!")
                    return
                
                risk_per_share = entry_price - stop_loss_price
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_loss_price, tag='LONG_ENTRY')
                    print(f"🌙🚀 VOLTAIC SURGE ACTIVATED! 🌩️\n"
                          f"Entry: {entry_price:.2f} | Size: {position_size} "
                          f"SL: {stop_loss_price:.2f} | Risk: {risk_per_share:.2f}")

        else:
            # Exit Conditions for Long
            if current_close <= self.kc_lower[-1]:
                self.position.close(tag='PROFIT_TAKE')
                print(f"🌙🎯 Profit Target Reached! Closing at {current_close:.2f}")
            elif current_close <= self.position.sl:
                self.position.close(t