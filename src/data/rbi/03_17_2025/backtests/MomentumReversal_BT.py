```python
import pandas as pd
import pandas_ta as ta
import talib
from backtesting import Strategy, Backtest
from backtesting.lib import crossover, crossunder

class MomentumReversal(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Clean data columns handled in data preprocessing
        
        # Calculate indicators using TA-Lib with self.I()
        self.fisher = self.I(self._calculate_fisher, name='Fisher')
        self.volume_accel = self.I(talib.ROC, self.data.Volume, timeperiod=14)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, 14)
        self.swing_high = self.I(talib.MAX, self.data.High, 20)
        self.swing_low = self.I(talib.MIN, self.data.Low, 20)
        
    def _calculate_fisher(self):
        # Calculate Fisher Transform using pandas_ta
        fisher = ta.fisher(self.data.High, self.data.Low, length=10)
        return fisher.iloc[:, 0]  # Return Fisher values column

    def next(self):
        if len(self.data) < 20:  # Ensure enough data for all indicators
            return

        # Exit conditions
        if self.position.is_long and crossunder(self.fisher[-3:], 2):
            self.position.close()
            print(f"🌕 Moon Dev Exit: LONG closed at {self.data.Close[-1]:.2f} | Profit: {self.position.pl_usd:.2f} 💰")
            
        if self.position.is_short and crossover(self.fisher[-3:], -2):
            self.position.close()
            print(f"🌑 Moon Dev Exit: SHORT closed at {self.data.Close[-1]:.2f} | Profit: {self.position.pl_usd:.2f} 💰")

        # Entry conditions
        if not self.position:
            # Long setup
            fisher_bullish = crossover(self.fisher[-3:], -2)
            vol_confirmation = self.volume_accel[-1] > self.volume_accel[-2] > 0
            price_action = self.data.Low[-1] <= self.swing_low[-1]
            
            if fisher_bullish and (self.adx[-1] < 25) and vol_confirmation and price_action:
                sl = self.swing_low[-1]
                self._enter_trade('long', sl)
                
            # Short setup
            fisher_bearish = crossunder(self.fisher[-3:], 2)
            vol_confirmation = self.volume_accel[-1] < self.volume_accel[-2] < 0
            price_action = self.data.High[-1] >= self.swing_high[-1]
            
            if fisher_bearish and (self.adx[-1] < 25) and vol_confirmation and price_action:
                sl = self.swing_high[-1]
                self._enter_trade('short', sl)
    
    def _enter_trade(self, direction, stop_loss):
        price = self.data.Close[-1]
        risk = abs(price - stop_loss)
        
        if risk == 0:
            return  # Avoid division by zero
        
        position_size = (self.broker.equity * self.risk_per_trade) / risk
        position_size = int(round(position_size))
        
        if position_size == 0:
            return  # Position too small
        
        if direction == 'long':
            self.buy(size=position_size, sl=stop_loss)
            print(f"🚀 Moon Dev LONG Entry: {price:.2f} | Size: {position_size} | SL: {stop_loss:.2f} 🌙")
        else:
            self.sell(size=position_size, sl=stop_loss)
            print(f"🌒 Moon Dev SHORT Entry: {price:.2f} | Size: {position_size} | SL: {stop_loss:.2f} 🌙")

# Data preprocessing
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
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