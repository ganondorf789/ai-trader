I'll help debug the code while maintaining the original strategy logic. Here are the key fixes needed:

1. Fixed incomplete print statement in short entry condition
2. Added proper position sizing calculation method
3. Fixed stop loss and take profit calculations
4. Added proper rounding for position sizes
5. Added missing code blocks

Here's the complete fixed code:

```python
# 🌙 Moon Dev's ConfluenceDivergence Backtest 🌙
import pandas as pd
from backtesting import Backtest, Strategy
import talib

# ====================
# DATA PREPARATION
# ====================
def load_data(path):
    data = pd.read_csv(
        path,
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
    return data

data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = load_data(data_path)

# ====================
# INDICATOR FUNCTIONS
# ====================
def calculate_ci(close, high, low, sma50_period=50, sma200_period=200, rsi_period=14, atr_period=14):
    """🌌 Calculate Confluence Index (CI) using TA-Lib"""
    sma50 = talib.SMA(close, sma50_period)
    sma200 = talib.SMA(close, sma200_period)
    rsi = talib.RSI(close, rsi_period)
    atr = talib.ATR(high, low, close, atr_period)
    ci = (sma50 - sma200) * rsi / (atr + 1e-6)  # Prevent division by zero
    return ci

# ====================
# CORE STRATEGY
# ====================
class ConfluenceDivergence(Strategy):
    risk_per_trade = 0.01  # 1% of equity per trade 🌕
    stop_loss_pct = 0.02  # 2% stop loss 🛡️
    take_profit_pct = 0.05  # 5% take profit 🎯
    
    def init(self):
        # Trend Indicators 🌗
        self.sma50 = self.I(talib.SMA, self.data.Close, timeperiod=50)
        self.sma200 = self.I(talib.SMA, self.data.Close, timeperiod=200)
        
        # Confluence Index System 🌠
        self.ci = self.I(calculate_ci, 
                        self.data.Close,
                        self.data.High,
                        self.data.Low)
        self.ci_signal = self.I(talib.SMA, self.ci, timeperiod=14)
        
        # Volatility Check 🌪️
        self.atr = self.I(talib.ATR, 
                         self.data.High,
                         self.data.Low,
                         self.data.Close, timeperiod=14)

    def calculate_size(self, entry_price, stop_price):
        """🌓 Calculate position size based on risk management rules"""
        risk_amount = self.equity * self.risk_per_trade
        risk_per_unit = abs(entry_price - stop_price)
        size = risk_amount / risk_per_unit
        return round(size)  # Round to whole units 🌙

    def next(self):
        current_price = self.data.Close[-1]
        
        # ====================
        # POSITION MANAGEMENT 🌙
        # ====================
        if not self.position:
            self.check_entries(current_price)
        else:
            self.check_exits(current_price)

    def check_entries(self, price):
        """🌓 Check for divergence entries with proper risk management"""
        # Long Entry: Bullish Divergence in Uptrend 🌟
        if (self.sma50[-1] > self.sma200[-1] and
            self.data.Low[-1] < self.data.Low[-2] and
            self.ci[-1] > self.ci[-2]):
            
            sl_price = price * (1 - self.stop_loss_pct)
            position_size = self.calculate_size(price, sl_price)
            if position_size > 0:
                print(f"🌙✨ BULLISH DIVERGENCE! Buying {position_size}