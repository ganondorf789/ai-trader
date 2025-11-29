I'll fix the code by removing all `backtesting.lib` imports and replacing the crossover function with the proper implementation. Here's the corrected version with Moon Dev themed debug prints:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()], errors='ignore')

# Column renaming and formatting
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class DynamicBreakoutMomentum(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Calculate all indicators using TA-Lib through self.I()
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=50, name='EMA 50')
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=200, name='EMA 200')
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, 14, name='ADX 14')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR 14')
        self.rsi5 = self.I(talib.RSI, self.data.Close, 5, name='RSI 5')
        self.recent_high = self.I(talib.MAX, self.data.High, 20, name='20-period High')
        
        self.highest_high = None  # For trailing stop calculation

    def next(self):
        # Wait for sufficient data
        if len(self.data) < 200:
            return

        # Entry conditions
        if not self.position:
            # EMA crossover condition (replaced backtesting.lib.crossover)
            golden_cross = (self.ema50[-2] < self.ema200[-2] and self.ema50[-1] > self.ema200[-1])
            
            # ADX strength condition
            adx_rising = self.adx[-1] > self.adx[-2] and self.adx[-1] > 25
            
            # Breakout condition
            price_breakout = self.data.Close[-1] > self.recent_high[-1]

            if golden_cross and adx_rising and price_breakout:
                # Calculate position size with proper rounding
                atr_value = self.atr[-1]
                if atr_value <= 0:
                    return
                
                risk_amount = self.equity * self.risk_percent
                stop_distance = 2 * atr_value
                position_size = risk_amount / stop_distance
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    self.highest_high = self.data.High[-1]  # Initialize trailing stop
                    print(f"🌕 MOON DEV ALERT: Long entry at {self.data.Close[-1]:.2f}!")
                    print(f"🚀 Size: {position_size} contracts | Risk: {self.risk_percent*100}% Equity")

        # Exit management
        else:
            # Update trailing stop
            self.highest_high = max(self.highest_high, self.data.High[-1])
            trail_stop = self.highest_high - 2 * self.atr[-1]
            
            # Trailing stop exit
            if self.data.Low[-1] < trail_stop:
                self.position.close()
                print(f"🌙 MOON DEV EXIT: Trailing stop triggered at {trail_stop:.2f}!")
                print(f"💎 Profit: {self.position.pl_pct:.2%} P/L")
            
            # RSI overbought exit
            elif self.rsi5[-1] > 70:
                self.position.close()
                print(f"🚨 MOON DEV EXIT: RSI {self.rsi5[-1]:.2f} overbought!")
                print(f"💎 Profit: {self.position.pl_pct:.2%} P/L")

# Run backtest