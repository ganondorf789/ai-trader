It seems there was some confusion in the initial response. Let me help you debug and complete the backtest code while maintaining the Moon Dev theme. Here's the fixed and complete version:

```python
# 🌙 Moon Dev Backtest AI Core Implementation
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌌 Cosmic Data Preparation
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

# 🌕 Data Cleansing Ritual
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class LunarPulseStrategy(Strategy):
    risk_per_trade = 0.02  # 🌕 2% Cosmic Risk Allocation
    rsi_period = 14
    sma_fast = 50
    sma_slow = 200
    swing_period = 20

    def init(self):
        # 🌠 Celestial Indicators Configuration
        self.sma_fast = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_fast, name='🌠 SMA50')
        self.sma_slow = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_slow, name='🌌 SMA200')
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period, name='📉 RSI14')
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period, name='🏔 Swing High')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period, name='🌊 Swing Low')

    def next(self):
        # 🌑 Wait for cosmic alignment (sufficient data)
        if len(self.data) < self.sma_slow:
            return

        # 🌕 Current Cosmic Readings
        price = self.data.Close[-1]
        equity = self.equity
        
        # 🌀 Momentum Vortex Detection
        sma_crossover = (self.sma_fast[-2] < self.sma_slow[-2] and 
                        self.sma_fast[-1] > self.sma_slow[-1])
        sma_crossunder = (self.sma_fast[-2] > self.sma_slow[-2] and 
                         self.sma_fast[-1] < self.sma_slow[-1])
        rsi_rising = self.rsi[-1] > self.rsi[-2]
        price_above_swing = price > self.swing_high[-1]

        # 🚀 Long Entry Protocol
        if not self.position:
            if (sma_crossover and 
                self.rsi[-1] < 70 and 
                rsi_rising and 
                price_above_swing):
                
                # 🌑 Calculate Stardust Stop Loss
                stop_loss = self.swing_low[-1]
                risk_amount = equity * self.risk_per_trade
                price_diff = abs(price - stop_loss)
                
                if price_diff == 0:
                    print("🌌 Cosmic Anomaly: Zero price difference detected!")
                    return
                
                position_size = risk_amount / price_diff
                position_size = int(round(position_size))  # 🌕 Round to whole units
                
                if position_size > 0:
                    take_profit = price + (2 * price_diff)
                    self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                    print(f"🚀🌕 LUNAR IGNITION! Buying {position_size} units at {price:.2f}")
                    print(f"   🌑 Stop Loss: {stop_loss:.2f}")
                    print(f"   🌕 Take Profit: {take_profit:.2f}")

        # 🌒 Exit Protocols
        elif self.position.is_long:
            if sma_crossunder or self.rsi[-1] > 70:
                self.position.close()
                print(f"🌒🌑 COSMIC RETREAT! Closing position at {price:.2f}")

# 🌟 Execute Lunar Backtest Ritual
bt = Backtest(data, LunarPulseStrategy, cash=1_000_000, exclusive_