```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION FOR VOLSURGEBREAKOUT STRATEGY 🚀

import pandas as pd
import talib
import pandas_ta as pdta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

class VolSurgeBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # 🌙 Clean and prepare data columns
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # 🚀 Calculate indicators using self.I()
        # Donchian Channel (20-period)
        self.donchian_upper = self.I(talib.MAX, self.data.High, 20, name='DONCHIAN_UPPER')
        self.donchian_lower = self.I(talib.MIN, self.data.Low, 20, name='DONCHIAN_LOWER')
        
        # Volume indicators (3-day surge vs 30-day median)
        self.volume_3d = self.I(talib.SUM, self.data.Volume, 288, name='VOL_3D')  # 3 days in 15m intervals
        self.volume_30d_median = self.I(pdta.median, self.data.Volume, 2880, name='VOL_30D_MED')  # 30 days
        
        # Keltner Channel (10-period)
        self.keltner_upper = self.I(pdta.keltner_channel_hband, 
                                   self.data.High, self.data.Low, self.data.Close, 
                                   length=10, scalar=1.5, name='KELTNER_UPPER')
        self.keltner_lower = self.I(pdta.keltner_channel_lband, 
                                   self.data.High, self.data.Low, self.data.Close, 
                                   length=10, scalar=1.5, name='KELTNER_LOWER')
        
        print("🌙 MOON DEV INDICATORS INITIALIZED! ✨")

    def next(self):
        current_close = self.data.Close[-1]
        donchian_upper = self.donchian_upper[-1]
        donchian_lower = self.donchian_lower[-1]
        volume_3d = self.volume_3d[-1]
        volume_30d_median = self.volume_30d_median[-1]
        keltner_upper = self.keltner_upper[-1]
        keltner_lower = self.keltner_lower[-1]

        # 🌙 MOON DEV RISK MANAGEMENT CALCULATIONS 🛡️
        if not self.position:
            # Long Entry: Close > Donchian Upper + Volume Surge 🌕
            if current_close > donchian_upper and volume_3d > 2 * volume_30d_median:
                sl = donchian_lower
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = current_close - sl
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.buy(size=position_size, sl=sl)
                        print(f"🌙 MOON DEV LONG ENTRY! ✨ Price: {current_close:.2f} | Size: {position_size} 🚀")
            
            # Short Entry: Close < Donchian Lower + Volume Surge 🌑
            elif current_close < donchian_lower and volume_3d > 2 * volume_30d_median:
                sl = donchian_upper
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = sl - current_close
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.sell(size=position_size, sl=sl)
                        print(f"🌙 MOON DEV SHORT ENTRY! ✨ Price: {current_close:.2f} | Size: {position_size} 🌪️")

        # 🌙 MOON DEV EXIT LOGIC 🎯
        if self.position.is_long and current_close < keltner_lower:
            self.position.close()
            print(f"🌙 LONG EXIT! 🌑 Price breached Keltner Lower: {current