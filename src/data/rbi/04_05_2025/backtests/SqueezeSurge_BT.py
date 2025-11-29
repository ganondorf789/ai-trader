```python
# 🌙 Moon Dev's SqueezeSurge Backtest Implementation 🚀

import pandas as pd
import talib
from backtesting import Backtest, Strategy

class SqueezeSurge(Strategy):
    risk_per_trade = 0.02  # 2% risk per trade 🌙
    
    def init(self):
        # 🌟 TTM Squeeze Components
        # Bollinger Bands (20,2)
        self.bb_upper = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0], 
                              self.data.Close, name='BB_Upper')
        self.bb_lower = self.I(lambda c: talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2], 
                              self.data.Close, name='BB_Lower')
        
        # Keltner Channel (20EMA + 2*ATR20)
        self.ema20 = self.I(talib.EMA, self.data.Close, 20, name='EMA20')
        self.atr20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 20, name='ATR20')
        self.kc_upper = self.I(lambda ema, atr: ema + 2*atr, self.ema20, self.atr20, name='KC_Upper')
        self.kc_lower = self.I(lambda ema, atr: ema - 2*atr, self.ema20, self.atr20, name='KC_Lower')
        
        # 📈 Volume Surge Indicator
        self.volume_90 = self.I(lambda v: v.rolling(20).quantile(0.9), 
                              self.data.Volume, name='Volume_90th')
        
        # 🛑 ATR Trailing Stop (14-period)
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR14')
        
        self.prev_squeeze = False
        self.trailing_stop = None
        self.highest_high = None

    def next(self):
        if len(self.data) < 20:  # Warmup period
            return
        
        # 🌀 Current Indicator Values
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        squeeze_on = (self.bb_upper[-1] < self.kc_upper[-1]) and \
                    (self.bb_lower[-1] > self.kc_lower[-1])
        
        # 🌙 Entry Logic
        if not self.position:
            if self.prev_squeeze and not squeeze_on:
                if current_volume > self.volume_90[-1]:
                    risk_amount = self.risk_per_trade * self.equity
                    atr_value = self.atr14[-1]
                    if atr_value == 0:
                        return
                    
                    position_size = int(round(risk_amount / (2 * atr_value)))
                    if position_size > 0:
                        self.buy(size=position_size)
                        self.highest_high = self.data.High[-1]
                        self.trailing_stop = self.highest_high - 2*self.atr14[-1]
                        print(f"🌙✨ MOON DEV LONG ENTRY 🚀 | Price: {current_close:.2f} | Size: {position_size}")

        # 🛑 Exit Logic
        if self.position.is_long:
            # Update trailing stop
            self.highest_high = max(self.highest_high, self.data.High[-1])
            new_stop = self.highest_high - 2*self.atr14[-1]
            self.trailing_stop = max(self.trailing_stop, new_stop) if self.trailing_stop else new_stop
            
            # Exit conditions
            if squeeze_on or self.data.Low[-1] < self.trailing_stop:
                self.position.close()
                print(f"🌙🛑 MOON DEV EXIT | Price: {current_close:.2f} | Reason: {'Squeeze Re-entry' if squeeze_on else 'Trailing Stop'}")

        self.prev_squeeze = squeeze_on

# 🚀 Data Preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
                   parse_dates=['datetime