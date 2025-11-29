```python
# 🌙 Moon Dev's FlowMomentum Backtest Implementation 🚀

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib

class FlowMomentum(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    swing_period = 20    # Swing low period
    obv_period = 20      # OBV smoothing period
    sma_period = 50      # Price SMA period
    entry_delay = 288    # 3 days in 15m intervals (288 periods)
    max_entry_window = 480  # 5 days (480 periods)

    def init(self):
        # 🌊 Calculate indicators using TA-Lib
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume)
        self.obv_ma = self.I(talib.SMA, self.obv, timeperiod=self.obv_period)
        self.price_sma = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period)
        
        # Track signal bar index
        self.signal_bar = -1

    def next(self):
        # 🌙 Check for new OBV crossover
        if crossover(self.obv, self.obv_ma):
            self.signal_bar = len(self.data) - 1
            print(f"✨ Potential Moon Signal Detected at {self.data.index[-1]}")

        # Check active signal status
        if self.signal_bar == -1:
            return

        bars_since_signal = len(self.data) - 1 - self.signal_bar
        
        # Expire signal if beyond 5 days
        if bars_since_signal > self.max_entry_window:
            self.signal_bar = -1
            return

        # Entry check after 3 days
        if bars_since_signal >= self.entry_delay and not self.position:
            # Get reference prices
            signal_price = self.data.Close[self.signal_bar]
            current_price = self.data.Close[-1]
            sma_signal = self.price_sma[self.signal_bar]
            sma_current = self.price_sma[-1]
            
            # 🚀 Entry conditions
            price_up = current_price > signal_price
            sma_up = sma_current > sma_signal
            obv_strong = self.obv[-1] > self.obv_ma[-1]
            
            if price_up and sma_up and obv_strong:
                # Calculate risk parameters
                stop_loss = self.swing_low[self.signal_bar]
                risk_per_share = current_price - stop_loss
                
                if risk_per_share <= 0:
                    return
                
                # Calculate position size
                risk_amount = self.equity * self.risk_percent
                position_size = int(round(risk_amount / risk_per_share))
                position_size = min(position_size, int(self.equity // current_price))
                
                if position_size > 0:
                    take_profit = current_price + 2*risk_per_share
                    self.buy(
                        size=position_size,
                        sl=stop_loss,
                        tp=take_profit,
                        tag="Moon Momentum Entry"
                    )
                    print(f"🚀🌙 LONG ENTRY | Size: {position_size} | Price: {current_price:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
                    self.signal_bar = -1

        # Exit conditions for open positions
        if self.position:
            # 🌘 Momentum reversal exit
            if crossover(self.obv_ma, self.obv):
                self.position.close()
                print(f"🌒🌊 Momentum Reversal Exit at {self.data.Close[-1]:.2f}")

# Data preparation 🌍
data = pd.read_csv("/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv")

# Clean and format data 🧹
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# Launch Moon Backtest