I'll fix the incomplete backtest code while maintaining the original strategy logic. Here's the complete fixed version with Moon Dev themed improvements:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class VolatilitySqueeze(Strategy):
    risk_percentage = 0.01  # 1% risk per trade
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col.lower()])
        
        # Calculate indicators using TA-Lib with self.I()
        # ATR and its MA
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR_14')
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=20, name='ATR_MA_20')
        
        # Bollinger Bands (using TA-Lib BBANDS)
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            lambda: talib.BBANDS(self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2),
            name=['BB_upper', 'BB_middle', 'BB_lower']
        )
        
        print("🌙 Moon Dev Indicators Initialized! 🚀✨")

    def next(self):
        current_close = self.data.Close[-1]
        current_atr = self.atr[-1]
        current_atr_ma = self.atr_ma[-1]
        current_bb_upper = self.bb_upper[-1]
        current_bb_middle = self.bb_middle[-1]
        
        # Moon Dev Debug Prints 🌙
        print(f"🌙 Current Close: {current_close} | ATR: {current_atr:.2f} < MA: {current_atr_ma:.2f} {'✅' if current_atr < current_atr_ma else '❌'} | BB_upper: {current_bb_upper:.2f} {'🚀' if current_close > current_bb_upper else '💤'}")

        # Entry Conditions
        if not self.position:
            # Condition 1: ATR < ATR MA
            atr_condition = current_atr < current_atr_ma
            
            # Condition 2: Close breaks above BB upper
            price_condition = current_close > current_bb_upper
            
            if atr_condition and price_condition:
                # Risk Management Calculation 🌙
                risk_amount = self.equity * self.risk_percentage
                risk_per_trade = 2 * current_atr
                position_size = int(round(risk_amount / risk_per_trade))
                
                print(f"🚀 MOON ENTRY SIGNAL! 🌟 Size: {position_size} @ {current_close}")
                self.buy(size=position_size, sl=current_close - 2*current_atr)
                print(f"🌙 Initial Stop Set: {current_close - 2*current_atr:.2f} ✨")

        # Exit Conditions
        if self.position:
            # Trailing Stop Logic 🌙
            trailing_stop = self.position.sl
            new_stop = max(trailing_stop, current_close - 2*current_atr)
            self.position.sl = new_stop

            # Exit on BB middle crossover (manual implementation)
            if current_close < current_bb_middle:
                print(f"🌙 REVERSAL EXIT! Closing @ {current_close} 🛑")
                self.position.close()
            elif self.data.Low[-1] < new_stop:
                print(f"🚀 TRAILING STOP HIT! Closing @ {current_close} 🛑")
                self.position.close()

# Data Preparation 🌙
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

print("🌙 MOON BACKTEST STARTING... 🚀✨