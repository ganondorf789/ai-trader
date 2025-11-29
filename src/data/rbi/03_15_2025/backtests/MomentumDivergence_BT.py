```python
# 🌙 Moon Dev's Momentum Divergence Backtest 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# 🧹 Data Cleaning & Preparation
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

class MomentumDivergence(Strategy):
    risk_per_trade = 0.01  # 1% account risk
    trailing_stop = True   # 🌙 Enable Moon Trailing Stops
    
    def init(self):
        # 🌠 Moon Indicators
        self.ema20 = self.I(talib.EMA, self.data.Close, 20, name='EMA20')
        self.ema50 = self.I(talib.EMA, self.data.Close, 50, name='EMA50')
        self.rsi = self.I(talib.RSI, self.data.Close, 14, name='RSI')
        
        # 🔍 Swing Lows Detection
        self.swing_low = self.I(talib.MIN, self.data.Low, 5, name='Swing Low')
        self.rsi_low = self.I(talib.MIN, self.rsi, 5, name='RSI Low')
        
        # 📊 Volume Filter
        self.vol_sma = self.I(talib.SMA, self.data.Volume, 96, name='Volume SMA')
        
        # 📅 Daily Loss Protection
        self.daily_equity = [self._broker.starting_cash]
        self.current_day = None

    def next(self):
        # ⚠️ Daily Risk Check
        current_date = self.data.index[-1].date()
        if current_date != self.current_day:
            self.current_day = current_date
            self.daily_equity.append(self.equity)
            if (self.daily_equity[-2] - self.daily_equity[-1])/self.daily_equity[-2] > 0.05:
                print(f"🌧️ MOON DEV ALERT: 5% Daily Loss Limit Triggered!")
                return

        if self.position:
            # 🛑 Exit Conditions
            price = self.data.Close[-1]
            entry = self.position.entry_price
            sl = self.position.sl
            risk = entry - sl
            
            # 🚀 Trailing Stop Logic
            if self.trailing_stop and price >= entry + risk:
                self.position.sl = entry
                print(f"🔒 MOON TRAILING: Moved SL to breakeven!")
            
            # 🎯 Take Profit Check
            if price >= entry + 2*risk:
                self.position.close()
                print(f"🎯 MOON PROFIT: 2:1 Reward Reached!")
                return
            
            # 📉 Trend Reversal Exit
            if crossover(self.ema50, self.ema20):
                self.position.close()
                print(f"📉 MOON EXIT: EMA Crossover Down")
                return
            
            # 🌡️ Overbought Exit
            if self.rsi[-1] > 70:
                self.position.close()
                print(f"🌡️ MOON EXIT: RSI Overbought")
                return
        else:
            # 🚦 Entry Conditions
            ema_cross = crossover(self.ema20, self.ema50)
            rsi_valid = self.rsi[-1] < 70
            vol_valid = (self.data.Volume[-1] > 0.8*self.vol_sma[-1]) and \
                      (self.data.Volume[-1] < 1.2*self.vol_sma[-1])
            
            # 🔄 Bullish Divergence Check
            div_valid = (self.swing_low[-1] < self.swing_low[-2]) and \
                       (self.rsi_low[-1] > self.rsi_low[-2])
            
            if ema_cross and rsi_valid and vol_valid and div_valid:
                # 🧮 Position Sizing
                sl_price = self.swing_low[-1]
                risk_share = self.data.Close[-1] - sl