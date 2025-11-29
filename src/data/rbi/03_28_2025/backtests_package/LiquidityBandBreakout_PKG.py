I'll fix the code by removing all backtesting.lib imports and replacing any related functions with proper alternatives. Here's the corrected version with Moon Dev themed debug prints:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Load and preprocess data
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class LiquidityBandBreakout(Strategy):
    def init(self):
        # Bollinger Band Width calculation using TA-Lib
        def calculate_bb_width(close):
            upper, middle, lower = talib.BBANDS(close, 
                                              timeperiod=20,
                                              nbdevup=2,
                                              nbdevdn=2)
            return (upper - lower) / middle
        
        self.bb_width = self.I(calculate_bb_width, self.data.Close, name='BB_WIDTH')
        
        # Swing high/low for liquidity clusters
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='SWING_HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='SWING_LOW')

    def next(self):
        # Moon Dev debug prints
        if len(self.data) % 100 == 0:
            print(f"🌙 Moon Dev Pulse | Bar {len(self.data)} | Price: {self.data.Close[-1]:.2f} | BB Width: {self.bb_width[-1]:.2f}")

        # Trade management
        if len(self.trades) >= 3:
            return

        current_close = self.data.Close[-1]
        current_bb_width = self.bb_width[-1]

        # Long entry logic
        if (current_bb_width < 0.5 and
            current_close > self.swing_high[-1] and
            not self.position):
            
            risk_amount = self.equity * 0.005  # 0.5% risk
            stop_loss = self.swing_low[-1]
            risk_distance = current_close - stop_loss
            
            if risk_distance <= 0:
                print(f"✨ Moon Dev Alert | Invalid SL for long @ {current_close:.2f}")
                return

            position_size = int(round(risk_amount / risk_distance))
            take_profit = current_close + (3 * risk_distance)  # 3:1 reward ratio
            
            if position_size > 0:
                self.buy(size=position_size, 
                        sl=stop_loss,
                        tp=take_profit,
                        tag="LiquidityBreakoutLong")
                print(f"🚀 Moon Dev LONG | Size: {position_size} | Entry: {current_close:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")

        # Short entry logic
        elif (current_bb_width < 0.5 and
              current_close < self.swing_low[-1] and
              not self.position):
            
            risk_amount = self.equity * 0.005
            stop_loss = self.swing_high[-1]
            risk_distance = stop_loss - current_close
            
            if risk_distance <= 0:
                print(f"✨ Moon Dev Alert | Invalid SL for short @ {current_close:.2f}")
                return

            position_size = int(round(risk_amount / risk_distance))
            take_profit = current_close - (3 * risk_distance)
            
            if position_size > 0:
                self.sell(size=position_size,
                         sl=stop_loss,
                         tp=take_profit,
                         tag="LiquidityBreakoutShort")
                print(f"🌑 Moon Dev SHORT | Size: {position_size} | Entry: {current_close:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")

        # Volatility expansion exit
        if current_bb_width > 1.5 and self.position:
            print(f"✨ Moon Dev Closing | BB Expansion {current_bb_width:.2f}")
            self.position.close()

# Initialize and run backtest
bt = Backtest(data, 
             LiquidityBandBreakout,