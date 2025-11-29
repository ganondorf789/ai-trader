```python
# 🌙 Moon Dev's VoltaicReversal Backtest Implementation
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib

# Clean and prepare data
def prepare_data(filepath):
    data = pd.read_csv(filepath, parse_dates=['datetime'], index_col='datetime')
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Drop unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    # Standardize column names
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    return data

# Load BTC data
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = prepare_data(data_path)

class VoltaicReversal(Strategy):
    risk_per_trade = 0.01  # 1% of equity per trade
    
    def init(self):
        # 🌙 Calculate indicators with proper TA-Lib integration
        # Stochastic Oscillator (14,3,3)
        self.stoch_k = self.I(lambda high, low, close: talib.STOCH(high, low, close, 
                               fastk_period=14, slowk_period=3, slowk_matype=0,
                               slowd_period=3, slowd_matype=0)[0],
                               self.data.High, self.data.Low, self.data.Close,
                               name='Stoch %K')
        
        self.stoch_d = self.I(lambda high, low, close: talib.STOCH(high, low, close,
                               fastk_period=14, slowk_period=3, slowk_matype=0,
                               slowd_period=3, slowd_matype=0)[1],
                               self.data.High, self.data.Low, self.data.Close,
                               name='Stoch %D')
        
        # Volatility indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        self.swing_high = self.I(talib.MAX, self.data.High, 20, name='Swing High')
        self.swing_low = self.I(talib.MIN, self.data.Low, 20, name='Swing Low')

    def next(self):
        # Wait for sufficient data
        if len(self.data) < 20:
            return
        
        # 🌟 Current indicator values
        current_k = self.stoch_k[-1]
        prev_k = self.stoch_k[-2] if len(self.stoch_k) >= 2 else current_k
        current_low = self.data.Low[-1]
        current_high = self.data.High[-1]
        swing_low = self.swing_low[-1]
        swing_high = self.swing_high[-1]
        atr = self.atr[-1]

        # Calculate position size
        equity = self.equity
        risk_amount = equity * self.risk_per_trade

        # 🚀 Long Entry: Stochastic crossover from oversold + swing low confirmation
        if not self.position.is_long:
            if (current_k > 20 and prev_k <= 20) and (current_low == swing_low):
                entry_price = self.data.Close[-1]
                stop_loss = swing_low - 1.5 * atr
                
                # Validate stop loss
                if stop_loss >= entry_price:
                    return  # Skip invalid stop loss
                
                risk_per_share = entry_price - stop_loss
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    take_profit = entry_price + 2 * (entry_price - stop_loss)
                    self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                    print(f"🌙✨🚀 MOON DEV LONG SIGNAL | Entry: {entry_price:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f} | Size: {position_size}")

        # 🌑 Short Entry: Stochastic crossover from overbought + swing high confirmation
        if not self.position.is_short:
            if (current_k < 80 and prev_k >= 80) and (current_high == swing_high):
                entry_price = self.data.Close[-1]
                stop_loss = swing_high + 1.