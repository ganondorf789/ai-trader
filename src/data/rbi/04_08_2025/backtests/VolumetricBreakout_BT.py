```python
# 🌙 Moon Dev's VolumetricBreakout Backtest Script
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib
import numpy as np

class VolumetricBreakout(Strategy):
    # Risk management parameters
    risk_per_trade = 0.01  # 1% of equity
    atr_multiplier = 1.5  # For trailing stop
    
    def init(self):
        # 🌙 Indicator Calculation Phase
        self.ema20 = self.I(talib.EMA, self.data.Close, timeperiod=20)
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.upper_band = self.I(lambda ema, atr: ema + 2*atr, self.ema20, self.atr14, name='Upper Band')
        self.lower_band = self.I(lambda ema, atr: ema - 2*atr, self.ema20, self.atr14, name='Lower Band')
        self.volume_ma20 = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        self.rsi14 = self.I(talib.RSI, self.data.Close, timeperiod=14)
        self.atr_ma20 = self.I(talib.SMA, self.atr14, timeperiod=20)
        
        print("✨ Moon Dev Indicators Activated! ✨")

    def next(self):
        # 🌙 Skip calculations until all indicators are ready
        if len(self.data) < 20:
            return

        # 🌙 Volatility Filter
        current_atr = self.atr14[-1]
        atr_ma = self.atr_ma20[-1]
        if current_atr < 0.5 * atr_ma:
            print("🌙 Volatility Too Low! Trading Suspended.")
            return

        # 🌙 Entry Logic
        if not self.position:
            # Long Entry Conditions
            if (self.data.Close[-1] > self.upper_band[-1] and
                self.data.Volume[-1] > self.volume_ma20[-1] and
                self.rsi14[-1] < 70):
                
                # Risk Management Calculation
                risk_amount = self.equity * self.risk_per_trade
                stop_loss = self.lower_band[-1]
                risk_per_share = self.data.Close[-1] - stop_loss
                
                if risk_per_share <= 0:
                    print("🌙 Invalid Stop Loss! Trade Canceled.")
                    return
                
                position_size = int(round(risk_amount / risk_per_share))
                if position_size == 0:
                    print("🌙 Position Size Too Small! Trade Canceled.")
                    return
                
                # Execute Buy Order
                self.buy(size=position_size, sl=stop_loss)
                print(f"🚀 MOON DEV LONG ENTRY! Size: {position_size} @ {self.data.Close[-1]}")
                
                # Initialize trailing stop
                self.trailing_stop = stop_loss

        # 🌙 Exit Logic
        if self.position.is_long:
            # RSI Exit
            if crossover(70, self.rsi14):
                self.position.close()
                print(f"🌙 RSI EXIT SIGNAL @ {self.data.Close[-1]}")
            
            # Trailing Stop Update
            recent_lows = self.data.Low[-3:]
            if len(recent_lows) >= 3:
                new_trailing = np.min(recent_lows)
                self.trailing_stop = max(self.trailing_stop, new_trailing)
                self.position.sl = self.trailing_stop
                print(f"🔧 Updating Moon Trail: {self.trailing_stop}")

# 🌙 Data Preparation Ritual
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
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

# 🌙 Launch Backtest Sequence
bt = Backtest(data, VolumetricBreakout, cash=1_000_000,