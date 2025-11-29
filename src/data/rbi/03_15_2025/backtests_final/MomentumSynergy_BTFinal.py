I notice the code is incomplete and has some issues that need to be fixed. Let me help you debug and complete it while maintaining the original strategy logic. Here's the fixed and complete version with Moon Dev-themed debug prints:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Data preparation and cleaning
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and format columns 🌙
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

class MomentumSynergy(Strategy):
    def init(self):
        # 🌕 Moon Dev Strategy Parameters ✨
        self.base_size = 0.01  # 1% equity
        self.current_size = self.base_size
        self.consecutive_wins = 0
        self.max_size = 0.05    # 5% cap
        self.max_consecutive = 5  # Max winning streak
        self.max_drawdown = 0.20  # 20% max drawdown
        
        # 🚀 TA-Lib Indicators using self.I() wrapper
        self.macd_line = self.I(lambda c: talib.MACD(c, 12, 26, 9)[0], self.data.Close)
        self.rsi = self.I(talib.RSI, self.data.Close, 14)
        self.correlation = self.I(talib.CORREL, self.macd_line, self.rsi, 14)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
        self.initial_equity = self._broker.starting_cash
        self.max_equity = self.initial_equity

    def next(self):
        # 🌙 Check for sufficient data
        if len(self.correlation) < 2 or len(self.atr) < 1:
            return
            
        current_corr = self.correlation[-1]
        prev_corr = self.correlation[-2]
        current_price = self.data.Close[-1]
        atr_value = self.atr[-1]

        # 🚨 Risk Management: Max 20% Drawdown
        current_drawdown = (self.max_equity - self.equity) / self.max_equity
        if current_drawdown >= self.max_drawdown:
            if self.position:
                self.position.close()
                print(f"🚨 MOON DEV EMERGENCY EXIT 🌑 | Max Drawdown Reached!")
            return

        # Update max equity
        self.max_equity = max(self.max_equity, self.equity)

        # 🌕 Entry Logic: Correlation crosses above 0.7
        if not self.position and current_corr > 0.7 and prev_corr <= 0.7:
            risk_amount = self.current_size * self.equity
            position_size = int(round(risk_amount / current_price))
            
            if position_size > 0:
                sl = current_price - 2 * atr_value
                tp = current_price + 4 * atr_value  # 2:1 RR
                self.buy(size=position_size, sl=sl, tp=tp)
                print(f"🌕 Moon Dev LONG Entry 🚀 | Size: {position_size} | Price: {current_price:.2f}")
                print(f"   SL: {sl:.2f} | TP: {tp:.2f} 🌙")

        # 🌑 Exit Logic: Correlation crosses below 0.7
        elif self.position.is_long and current_corr < 0.7 and prev_corr >= 0.7:
            self.position.close()
            print(f"🌑 Moon Dev EXIT Signal ✨ | Price: {current_price:.2f}")

            # Update position sizing based on trade outcome
            if self.trades:
                last_trade = self.trades[-1]
                if last_trade.pnl > 0:
                    self.current_size = min(self.current_size * 1.2, self.max_size)
                    self.consecutive_wins += 1
                    print(f"✨ WINNER! Size ↑ {self.current_size*100:.2f}%