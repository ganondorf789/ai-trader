Here's the fully debugged and Moon Dev themed VolSqueezeBreakout strategy code:

```python
# 🌙 Moon Dev's VolSqueezeBreakout Backtest Script 🚀✨

import pandas as pd
import talib
import numpy as np
from backtesting import Strategy, Backtest

# Data Preparation 🌙
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean and format columns ✨
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolSqueezeBreakout(Strategy):
    risk_per_trade = 0.01  # 1% equity risk per trade 🌙
    
    def init(self):
        # Bollinger Bands (20,2) with TA-Lib ✨
        self.bb_upper = self.I(self._calc_bb_upper)
        self.bb_lower = self.I(self._calc_bb_lower)
        
        # Volume MA and ATR 🚀
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
    def _calc_bb_upper(self, close):
        upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        return upper
        
    def _calc_bb_lower(self, close):
        _, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        return lower

    def next(self):
        # Wait for sufficient data 🌙
        if len(self.data) < 20:
            return
            
        # Current market conditions ✨
        close = self.data.Close[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        volume = self.data.Volume[-1]
        volume_ma = self.volume_ma[-1]
        atr = self.atr[-1]

        # Strategy logic 🚀
        bb_squeeze = ((upper - lower) / upper) <= 0.01 if upper != 0 else False
        volume_surge = volume >= 2.5 * volume_ma if volume_ma != 0 else False
        
        # Entry signals 🌙
        if not self.position:
            risk_amount = self.equity * self.risk_per_trade
            
            if bb_squeeze and volume_surge:
                if close > upper:
                    # Long entry ✨
                    sl = close - atr
                    size = int(round(risk_amount / (close - sl)))
                    if size > 0:
                        self.buy(size=size, sl=sl)
                        self.trailing_peak = self.data.High[-1]
                        print(f"🌙 MOON DEV LONG ENTRY! Size: {size} ✨ | Price: {close:.2f} 🚀")
                        
                elif close < lower:
                    # Short entry 🚀
                    sl = close + atr
                    size = int(round(risk_amount / (sl - close)))
                    if size > 0:
                        self.sell(size=size, sl=sl)
                        self.trailing_trough = self.data.Low[-1]
                        print(f"🌙 MOON DEV SHORT ENTRY! Size: {size} ✨ | Price: {close:.2f} 🚀")

        # Exit management 🌙
        else:
            if self.position.is_long:
                # Update trailing peak ✨
                self.trailing_peak = max(self.trailing_peak, self.data.High[-1])
                trail_stop = self.trailing_peak - 2*atr
                
                if close < upper or self.data.Low[-1] < trail_stop:
                    self.position.close()
                    print(f"🌙 MOON DEV LONG EXIT! ✨ | Price: {close:.2f} 🚀")
                    
            elif self.position.is_short:
                # Update trailing trough 🚀
                self.trailing_trough = min(self.trailing