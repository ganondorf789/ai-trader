```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib
import numpy as np

class DynamicBreakout(Strategy):
    atr_multiplier = 1.5
    atr_period = 14
    sma_period = 20
    rsi_period = 14
    volume_sma_period = 5
    time_exit_bars = 10
    risk_pct = 0.01

    def init(self):
        # Calculate core indicators 🌙
        self.sma = self.I(talib.SMA, self.data.Close, self.sma_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.upper_band = self.I(lambda sma, atr: sma + self.atr_multiplier * atr, self.sma, self.atr)
        self.lower_band = self.I(lambda sma, atr: sma - self.atr_multiplier * atr, self.sma, self.atr)
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, self.volume_sma_period)
        self.entry_bar = 0

    def next(self):
        if len(self.data.Close) < 2:
            return

        prev_close = self.data.Close[-2]
        upper_prev = self.upper_band[-2]
        lower_prev = self.lower_band[-2]
        rsi_prev = self.rsi[-2]
        volume_prev = self.data.Volume[-2]
        volume_sma_prev = self.volume_sma[-2]

        if not self.position:
            # Long entry constellation 🌠
            if prev_close > upper_prev and rsi_prev > 50 and volume_prev > volume_sma_prev:
                self.enter_trade(direction='long')
            
            # Short entry constellation 🌌
            elif prev_close < lower_prev and rsi_prev < 50 and volume_prev > volume_sma_prev:
                self.enter_trade(direction='short')
        else:
            # Time-based exit rocket launch 🚀
            if (len(self.data) - 1 - self.entry_bar) >= self.time_exit_bars:
                print(f"⏰ Moon Dev Time Exit! Closing after {self.time_exit_bars} bars")
                self.position.close()

    def enter_trade(self, direction):
        entry_price = self.data.Open[-1]
        atr_value = self.atr[-2]
        
        if direction == 'long':
            sl = entry_price - atr_value
            tp = entry_price + 2 * atr_value
        else:
            sl = entry_price + atr_value
            tp = entry_price - 2 * atr_value

        risk_amount = self.risk_pct * self.equity
        risk_per_share = abs(entry_price - sl)
        
        if risk_per_share == 0:
            return

        position_size = int(round(risk_amount / risk_per_share))
        
        if position_size > 0:
            if direction == 'long':
                self.buy(size=position_size, sl=sl, tp=tp)
                print(f"🌕 BUY Signal! Size: {position_size} | Entry: {entry_price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
            else:
                self.sell(size=position_size, sl=sl, tp=tp)
                print(f"🌑 SELL Signal! Size: {position_size} | Entry: {entry_price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
            
            self.entry_bar = len(self.data) - 1

# Prepare cosmic data 🌍
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean data columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to celestial coordinates 🌌
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

data['datetime'] = pd.to_datetime(data['datetime'])