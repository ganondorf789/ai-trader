```python
# 🌙 Moon Dev's Squeeze Momentum Backtest 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossval

# Data preprocessing moon magic ✨
def load_data(path):
    data = pd.read_csv(path)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Drop unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    # Map columns to backtesting requirements
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    data.index = pd.to_datetime(data['datetime'])
    return data

data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = load_data(data_path)

class SqueezeMomentum(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # 🌗 Bollinger Bands Setup
        self.bb_upper = self.I(lambda x: talib.BBANDS(x, 20, 2, 2, 0)[0], self.data.Close)
        self.bb_lower = self.I(lambda x: talib.BBANDS(x, 20, 2, 2, 0)[2], self.data.Close)
        
        # 🌓 Bollinger Band Width Calculation
        self.bb_width = self.I(lambda x: 
            ((talib.BBANDS(x, 20, 2, 2, 0)[0] - talib.BBANDS(x, 20, 2, 2, 0)[2]) /
            talib.BBANDS(x, 20, 2, 2, 0)[1], self.data.Close)
        
        # 🌔 20-period SMA of BB Width
        self.bb_width_sma = self.I(talib.SMA, self.bb_width, 20)
        
        # 🌕 Volume-Weighted MACD (Using volume as input series)
        self.macd_line = self.I(lambda x: talib.MACD(x, 12, 26, 9)[0], self.data.Volume)
        
        # 🌑 ATR for Risk Management
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
        self.peak_equity = self._broker.starting_cash
        print("🌙 Moon Dev Indicators Activated! ✨")

    def next(self):
        # 🌗 Account Protection System
        current_equity = self._broker.equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            
        if current_equity < 0.95 * self.peak_equity:
            print(f"🌑 MOON DEV ALERT: 5% Drawdown Protection Engaged! Current Equity: ${current_equity:,.2f}")
            return
            
        # 🚀 Position Management
        open_trades = [t for t in self.trades if t.is_open]
        if len(open_trades) >= 3:
            return

        # 🌕 Long Entry Logic
        if not self.position and \
            self.bb_width[-1] < 0.5 * self.bb_width_sma[-1] and \
            self.data.Close[-1] > self.bb_upper[-1] and \
            crossover(self.macd_line, 0):
            
            atr_value = self.atr[-1]
            if atr_value == 0: return
            
            stop_loss = self.data.Close[-1] - atr_value
            risk_amount = self.risk_per_trade * current_equity
            position_size = int(round(risk_amount / (self.data.Close[-1] - stop_loss)))
            
            if position_size > 0:
                self.buy(size=position_size, 
                        sl=stop_loss,
                        tp=self.data.Close[-1] + 3*atr_value)
                print(f"🚀 MOON DEV LONG SIGNAL 🌕 | Size: {position_size} | Entry: {self.data.Close[-1]:.2f} | SL: {stop_loss:.2f} | TP: {self.data.Close[-1]+3*atr_value:.2f}")

        # 🌑 Short Entry Logic
        elif not self.position and \