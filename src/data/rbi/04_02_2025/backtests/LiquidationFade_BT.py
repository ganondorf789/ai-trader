```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Load and preprocess data
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean column names
data.columns = data.columns.str.strip().str.lower()

# Drop unnamed columns
unnamed_cols = [col for col in data.columns if 'unnamed' in col]
data = data.drop(columns=unnamed_cols)

# Rename and validate columns
column_mapping = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}
data.rename(columns=column_mapping, inplace=True)

required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"Missing required column: {col}")

if 'openinterest' not in data.columns:
    raise ValueError("Data must contain 'openinterest' column for this strategy")

class LiquidationFade(Strategy):
    risk_percent = 1  # 1% risk per trade
    
    def init(self):
        # Calculate indicators
        close_prices = self.data.Close
        
        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = talib.BBANDS(
            close_prices, timeperiod=20, nbdevup=3, nbdevdn=3
        )
        self.bb_upper_ind = self.I(lambda: self.bb_upper, name='BB_UPPER')
        self.bb_lower_ind = self.I(lambda: self.bb_lower, name='BB_LOWER')
        
        # Open Interest Momentum
        self.oi_roc = self.I(talib.ROC, self.data['openinterest'], timeperiod=1, name='OI_ROC')
        
        # RSI
        self.rsi = self.I(talib.RSI, close_prices, timeperiod=14, name='RSI')
        
        print("🌙 Moon Dev Indicators Initialized! ✨")

    def next(self):
        price = self.data.Close[-1]
        print(f"🌙 Current Price: {price:.2f} | RSI: {self.rsi[-1]:.1f} | OI Change: {self.oi_roc[-1]:.2f}%")
        
        if self.position:
            # Exit logic
            if self.position.is_short and crossover(30, self.rsi):
                self.position.close()
                print(f"🌑 Moon Dev Short Exit: RSI Cross Below 30 🌟")
                
            elif self.position.is_long and crossover(self.rsi, 70):
                self.position.close()
                print(f"🌕 Moon Dev Long Exit: RSI Cross Above 70 🌟")
        else:
            # Short Entry Logic
            if price > self.bb_upper_ind[-1] and self.oi_roc[-1] < -5:
                swing_high = talib.MAX(self.data.High, 20)[-1]
                stop_loss = swing_high * 1.01
                risk_amount = self.equity * self.risk_percent / 100
                risk_per_unit = stop_loss - price
                
                if risk_per_unit > 0:
                    size = int(round(risk_amount / risk_per_unit))
                    self.sell(size=size, sl=stop_loss)
                    print(f"🌑 MOON DEV SHORT ENTRY 🌙 Size: {size} | SL: {stop_loss:.2f}")
            
            # Long Entry Logic
            elif price < self.bb_lower_ind[-1] and self.oi_roc[-1] < -5:
                swing_low = talib.MIN(self.data.Low, 20)[-1]
                stop_loss = swing_low * 0.99
                risk_amount = self.equity * self.risk_percent / 100
                risk_per_unit = price - stop_loss
                
                if risk_per_unit > 0:
                    size = int(round(risk_amount / risk_per_unit))
                    self.buy(size=size, sl=stop_loss)
                    print(f"🌕 MOON DEV LONG ENTRY 🌙 Size: {size} | SL: {stop_loss:.2f}")

# Run backtest
bt = Backtest(data, LiquidationFade, cash=1_000_000, trade_on_close=True)
stats =