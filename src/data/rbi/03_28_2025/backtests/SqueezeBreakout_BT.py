```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR SQUEEZEBREAKOUT STRATEGY 🚀

import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# =====================
# DATA PREPARATION 🌌
# =====================
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# =====================
# STRATEGY CLASS 🌗
# =====================
class SqueezeBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌕
    
    def init(self):
        # =====================
        # INDICATORS SETUP 📈
        # =====================
        # Bollinger Bands (20-period, 1.5σ)
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            lambda close: talib.BBANDS(close, timeperiod=20, nbdevup=1.5, nbdevdn=1.5),
            self.data.Close,
            name=['BB_upper', 'BB_middle', 'BB_lower']
        )
        
        # ADX (10-period)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=10, name='ADX')
        
        # Stochastic Oscillator (5-3-3)
        self.stoch_k, self.stoch_d = self.I(
            lambda h,l,c: talib.STOCH(h, l, c, 
                                     fastk_period=5, slowk_period=3, 
                                     slowk_matype=0, slowd_period=3),
            self.data.High, self.data.Low, self.data.Close,
            name=['Stoch_K', 'Stoch_D']
        )
        
        # Swing High/Low (20-period)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='Swing_High')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='Swing_Low')

    def next(self):
        # =====================
        # ENTRY LOGIC 🚪
        # =====================
        if not self.position:
            # Check Bollinger Squeeze Condition
            prev_in_squeeze = (self.bb_upper[-2] - self.bb_lower[-2])/self.bb_middle[-2] < 0.02
            
            # Breakout Detection
            current_close = self.data.Close[-1]
            breakout_above = current_close > self.bb_upper[-1]
            breakout_below = current_close < self.bb_lower[-1]
            
            # ADX Trend Strength
            adx_trigger = self.adx[-1] > 30 and self.adx[-1] > self.adx[-2]
            
            if prev_in_squeeze and adx_trigger:
                if breakout_above:
                    # 🌙 LONG ENTRY SIGNAL 🌟
                    stop_price = self.swing_low[-1]
                    risk = current_close - stop_price
                    position_size = int(round((self.equity * self.risk_per_trade) / risk))
                    
                    if position_size > 0:
                        self.buy(size=position_size, sl=stop_price)
                        print(f"🌙✨ MOON DEV LONG! 🚀 Entry: {current_close:.2f} | Size: {position_size} | SL: {stop_price:.2f}")
                        
                elif breakout_below:
                    # 🌑 SHORT ENTRY SIGNAL 🌪️
                    stop_price = self.swing_high[-1]
                    risk = stop_price - current_close
                    position_size = int(round((self.equity * self.risk_per_trade) / risk))
                    
                    if position_size > 0:
                        self.sell(size=position_size, sl=stop_price)
                        print(f"🌙✨ MOON DEV SHORT! 🌪️ Entry: {