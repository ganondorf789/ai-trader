Here's the complete backtesting implementation for the VolatilityEdge strategy:

```python
# 🌙 Moon Dev Backtest Engine v1.0 🚀
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
import pandas as pd
import talib
import numpy as np

class VolatilityEdge(Strategy):
    # Strategy parameters
    volatility_period = 14  # ATR period for volatility measurement
    swing_window = 20       # Swing high/low detection window
    risk_per_trade = 0.01   # 1% risk per trade
    
    def init(self):
        # 🌙 Calculate volatility and swing levels using TA-Lib
        self.atr = self.I(talib.ATR, 
                         self.data.High, 
                         self.data.Low, 
                         self.data.Close, 
                         timeperiod=self.volatility_period,
                         name='🌙 Volatility (ATR)')
        
        self.swing_high = self.I(talib.MAX, 
                                self.data.High, 
                                timeperiod=self.swing_window,
                                name='🚀 Swing High')
        
        self.swing_low = self.I(talib.MIN, 
                               self.data.Low, 
                               timeperiod=self.swing_window,
                               name='🎯 Swing Low')
        
        print("🌙✨ Strategy Indicators Initialized!")

    def next(self):
        current_close = self.data.Close[-1]
        current_atr = self.atr[-1]
        equity = self.equity  # Current portfolio value
        
        # 🌙✨ Moon Dev Trading Logic
        if not self.position:
            # Long Entry: High Volatility Breakout
            if current_atr > 50 and crossover(self.data.Close, self.swing_high):
                stop_loss = self.swing_low[-1]
                risk_amount = self.risk_per_trade * equity
                risk_per_unit = current_close - stop_loss
                
                if risk_per_unit > 0:
                    position_size = int(round(risk_amount / risk_per_unit))
                    if position_size > 0:
                        self.buy(size=position_size, 
                                sl=stop_loss,
                                tag="🌙 HighVol Long")
                        print(f"🚀🚀 LONG ENTRY | Size: {position_size} | ATR: {current_atr:.2f} | Risk/Reward: 1:{((current_close - stop_loss)/current_atr):.1f}")
            
            # Short Entry: Low Volatility Breakdown
            elif current_atr < 20 and crossunder(self.data.Close, self.swing_low):
                stop_loss = self.swing_high[-1]
                risk_amount = self.risk_per_trade * equity
                risk_per_unit = stop_loss - current_close
                
                if risk_per_unit > 0:
                    position_size = int(round(risk_amount / risk_per_unit))
                    if position_size > 0:
                        self.sell(size=position_size,
                                 sl=stop_loss,
                                 tag="🌙 LowVol Short")
                        print(f"🎯🎯 SHORT ENTRY | Size: {position_size} | ATR: {current_atr:.2f} | Risk/Reward: 1:{((stop_loss - current_close)/current_atr):.1f}")
        else:
            # 🌙 Exit Conditions
            if self.position.is_long and current_atr < 40:
                self.position.close()
                print(f"✨ Closing LONG | ATR Reversion: {current_atr:.2f} < 40")
                
            if self.position.is_short and current_atr > 25:
                self.position.close()
                print(f"✨ Closing SHORT | ATR Reversion: {current_atr:.2f} > 25")

# 🌙 Data Preparation Ritual
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
ohlc = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# 🧹 Data Cleansing Spell
ohlc.columns = ohlc.columns.str.strip().str.lower()
ohlc = ohlc.drop(columns=[col for col in ohlc.columns if 'unnamed' in col])
ohlc = ohlc.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

print("🌙✨ Data Preparation Complete! First 3 candles:")
print(ohlc.head(3))

# 🚀 Launch Backtest
bt = Backtest(ohlc, VolatilityEdge, cash=1_000_000, commission=.002)
stats = bt.run()

print