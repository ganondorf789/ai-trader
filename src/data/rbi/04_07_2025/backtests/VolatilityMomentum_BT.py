```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR VOLATILITYMOMENTUM STRATEGY 🚀

import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class VolatilityMomentum(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌙 CALCULATE INDICATORS WITH SELF.I() WRAPPER
        self.cmo = self.I(talib.CMO, self.data.Close, timeperiod=14)
        
        # Bollinger Bands components
        def bb_upper(close, timeperiod, nbdev):
            upper, _, _ = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdev, nbdevdn=nbdev, matype=0)
            return upper
        self.upper_band = self.I(bb_upper, self.data.Close, 20, 2)
        
        def bb_lower(close, timeperiod, nbdev):
            _, _, lower = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdev, nbdevdn=nbdev, matype=0)
            return lower
        self.lower_band = self.I(bb_lower, self.data.Close, 20, 2)
        
        self.band_width = self.I(lambda u, l: u - l, self.upper_band, self.lower_band)
        self.band_width_low = self.I(talib.MIN, self.band_width, timeperiod=20)
        
        # Volatility stop calculations
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        print("🌙✨ MOON DEV INDICATORS INITIALIZED! READY FOR LIFTOFF! 🚀")

    def next(self):
        # 🌙 STRATEGY LOGIC EXECUTION
        if not self.position:
            if len(self.cmo) < 2 or len(self.band_width_low) < 1:
                return  # Insufficient data
            
            current_cmo = self.cmo[-1]
            prev_cmo = self.cmo[-2]
            current_width = self.band_width[-1]
            width_low = self.band_width_low[-1]
            
            # 🌙 ENTRY CONDITIONS
            cmo_cross = prev_cmo <= 50 and current_cmo > 50
            width_condition = current_width < width_low
            
            if cmo_cross and width_condition:
                # 🌙 RISK MANAGEMENT CALCULATIONS
                risk_amount = self.equity * self.risk_percent
                atr_value = self.atr[-1]
                
                if atr_value == 0:
                    print("🌙⚠️ ZERO ATR DETECTED! ABORTING TRADE! ⚠️")
                    return
                
                sl_price = self.data.Close[-1] - 1.5 * atr_value
                risk_per_share = abs(self.data.Close[-1] - sl_price)
                
                if risk_per_share == 0:
                    print("🌙⚠️ ZERO RISK PER SHARE! TRADE SKIPPED! ⚠️")
                    return
                
                position_size = int(round(risk_amount / risk_per_share))
                max_possible = int(self.equity // self.data.Close[-1])
                position_size = min(position_size, max_possible)
                
                if position_size > 0:
                    tp_price = self.data.Close[-1] + 2 * atr_value
                    self.buy(size=position_size, 
                            sl=sl_price,
                            tp=tp_price,
                            tag="MOON_DEV_LONG")
                    
                    print(f"🌙✨🚀 MOON DEV TRADE LAUNCHED! 🚀✨")
                    print(f"Entry: {self.data.Close[-1]:.2f} | Size: {position_size} units")
                    print(f"SL: {sl_price:.2f} | TP: {tp_price:.2f} | ATR: {atr_value:.2f}\n")
        else:
            # 🌙 MOON DEV TRAILING STOP LOGIC CAN BE ADDED HERE
            pass

# 🌙 DATA PREPARATION FOR MOON MISSION
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names like a Moon Dev pro 🌙
data.columns = data.columns.str.strip