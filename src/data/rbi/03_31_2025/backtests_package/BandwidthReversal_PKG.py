Here's the fixed code with all backtesting.lib imports and functions removed, replaced with proper Moon Dev themed implementations:

```python
# 🌙 Moon Dev's BandwidthReversal Backtest 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class BandwidthReversal(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # 🌟 Calculate Indicators using TA-Lib
        # Bollinger Bands Components
        self.bb_upper = self.I(lambda close: talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0], self.data.Close)
        self.bb_middle = self.I(lambda close: talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[1], self.data.Close)
        self.bb_lower = self.I(lambda close: talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2], self.data.Close)
        
        # Bollinger Band Width (BBW)
        self.bbw = self.I(lambda u, l, m: (u - l) / m, self.bb_upper, self.bb_lower, self.bb_middle)
        self.bbw_low = self.I(talib.MIN, self.bbw, timeperiod=20)  # 20-day BBW low
        
        # MACD Histogram
        self.macd_hist = self.I(lambda close: talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[2], self.data.Close)
        
        # ATR for volatility & position sizing
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=10)
        
        print("🌌 Moon Dev Indicators Initialized! BBW | MACD | ATR 🌠")

    def next(self):
        if not self.position:
            # 🚀 Check Entry Conditions
            current_bbw = self.bbw[-1]
            current_bbw_min = self.bbw_low[-1]
            macd_cross = (self.macd_hist[-2] < 0 and self.macd_hist[-1] > 0)  # Replaced crossover with direct comparison
            
            if current_bbw <= current_bbw_min and macd_cross:
                # 🌙 Calculate Position Size
                equity = self.broker.get_value()
                atr_value = self.atr[-1]
                risk_amount = equity * self.risk_percent
                position_size = int(round(risk_amount / atr_value))
                
                if position_size > 0:
                    entry_price = self.data.Close[-1]
                    sl_price = entry_price - atr_value * 1  # 1x ATR stop
                    tp_price = entry_price + atr_value * 2  # 2x ATR target
                    
                    self.buy(size=position_size, sl=sl_price, tp=tp_price)
                    print(f"🌙✨🚀 LONG Entry @ {entry_price:.2f} | Size: {position_size} | SL: {sl_price:.2f} | TP: {tp_price:.2f}")

    def notify_trade(self, trade):
        if trade.is_closed:
            emoji = "🚀🌙✨" if trade.pl > 0 else "💥🌧️"
            print(f"{emoji} Trade Closed | PnL: ${trade.pl:.2f} ({trade.pl_pct:.2%})")

# 🌕 Data Preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# 🧹 Clean & Format Data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

print("🔄🌙 Data Loaded & Cleaned | Columns:", data.columns.tolist())

# 🚀 Run Backtest
bt = Backtest(data, BandwidthReversal,