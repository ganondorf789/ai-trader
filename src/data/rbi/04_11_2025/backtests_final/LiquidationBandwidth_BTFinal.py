I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete debugged version with Moon Dev improvements:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Data preparation
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and prepare data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class LiquidationBandwidth(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    max_concurrent_trades = 3
    
    def init(self):
        # Core indicators 🌙
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0]
        self.bb_middle = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1]
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2]
        self.bb_width = self.I(lambda u, m, l: (u-l)/m, self.bb_upper, self.bb_middle, self.bb_lower)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.liq_long = self.I(talib.MIN, self.data.Low, timeperiod=20)
        self.liq_short = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.ma50 = self.I(talib.SMA, self.data.Close, timeperiod=50)
        
        # Daily tracking ✨
        self.current_day = None
        self.daily_initial_equity = self._broker._cash

    def _is_bb_low_vol(self):
        """Check if Bollinger Band width is in low volatility state"""
        current_width = self.bb_width[-1]
        width_range = np.percentile(self.bb_width, [0, 100])
        threshold = width_range[0] + 0.2 * (width_range[1] - width_range[0])
        return current_width <= threshold

    def _bullish_momentum(self):
        """Check for bullish momentum conditions"""
        return (self.data.Close[-1] > self.data.Open[-1] and 
                self.data.Close[-2] > self.data.Open[-2] and
                self.data.Close[-1] > (self.data.High[-1] + self.data.Low[-1])/2)

    def _bearish_momentum(self):
        """Check for bearish momentum conditions"""
        return (self.data.Close[-1] < self.data.Open[-1] and 
                self.data.Close[-2] < self.data.Open[-2] and
                self.data.Close[-1] < (self.data.High[-1] + self.data.Low[-1])/2)

    def _execute_trade(self, direction, price, sl, tp):
        """Execute trade with proper position sizing"""
        risk_amount = self.equity * self.risk_per_trade
        risk_per_unit = abs(price - sl)
        
        if risk_per_unit <= 0:
            print("🌑 BLACK HOLE ALERT: Invalid stop loss distance!")
            return
            
        position_size = round(risk_amount / risk_per_unit)
        
        if direction == 'LONG':
            self.buy(size=position_size, sl=sl, tp=tp)
            print(f"🚀 LAUNCHING LONG | Size: {position_size} | Entry: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
        else:
            self.sell(size=position_size, sl=sl, tp=tp)
            print(f"🌘 ENTERING SHORT | Size: {position_size} | Entry: {price:.2f} | SL: {sl:.2f} |