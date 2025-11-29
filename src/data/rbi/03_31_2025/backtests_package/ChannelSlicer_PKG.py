Here's the fixed code with all backtesting.lib imports removed and replaced with proper alternatives, along with Moon Dev themed debug prints:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import pandas_ta as ta
import talib
import numpy as np

# Data preparation
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
data['DateTime'] = pd.to_datetime(data['datetime'])
data.set_index('DateTime', inplace=True)
data.drop(columns=['datetime'], inplace=True, errors='ignore')

class ChannelSlicer(Strategy):
    def init(self):
        self.donchian_upper = self.I(talib.MAX, self.data.High, 20, name='Donchian Upper 🌙')
        self.donchian_lower = self.I(talib.MIN, self.data.Low, 20, name='Donchian Lower 🌙')
        self.volume_avg = self.I(talib.SMA, self.data.Volume, 20, name='Volume SMA ✨')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR 🚀')
        self.ci = self.I(self._calculate_ci, name='Choppiness Index 🌊')
        
    def _calculate_ci(self):
        return ta.ci(high=self.data.High, low=self.data.Low, close=self.data.Close, length=14)
    
    def next(self):
        if len(self.data.Close) < 20:
            return
        
        current_close = self.data.Close[-1]
        ci_value = self.ci[-1]
        donchian_upper = self.donchian_upper[-1]
        donchian_lower = self.donchian_lower[-1]
        volume = self.data.Volume[-1]
        volume_avg = self.volume_avg[-1]

        print(f"🌙 Moon Dev Status | Close: {current_close:.2f} | CI: {ci_value:.1f} | Vol: {volume:.2f}/{volume_avg:.2f}")

        if not self.position:
            if current_close > donchian_upper and ci_value < 35 and volume > volume_avg:
                self._enter_trade('long')
            elif current_close < donchian_lower and ci_value < 35 and volume > volume_avg:
                self._enter_trade('short')
                
        self._manage_exits()

    def _enter_trade(self, direction):
        box_size = self.atr[-1]
        if box_size <= 0:
            return

        risk_pct = 0.01
        risk_amount = self.equity * risk_pct
        risk_per_share = 2 * box_size
        position_size = int(round(risk_amount / risk_per_share))

        if position_size > 0:
            entry_price = self.data.Close[-1]
            if direction == 'long':
                self.buy(size=position_size, tag={
                    'direction': 'long',
                    'box_size': box_size,
                    'high_watermark': entry_price,
                    'entry_price': entry_price
                })
                print(f"🚀🌙 Moon Dev LONG Entry! {entry_price:.2f} | Size: {position_size}")
            else:
                self.sell(size=position_size, tag={
                    'direction': 'short',
                    'box_size': box_size,
                    'low_watermark': entry_price,
                    'entry_price': entry_price
                })
                print(f"🚀🌙 Moon Dev SHORT Entry! {entry_price:.2f} | Size: {position_size}")

    def _manage_exits(self):
        for trade in self.trades:
            tag = trade.tag
            box_size = tag['box_size']
            entry = tag['entry_price']

            if tag['direction'] == 'long':
                current_high = self.data.High[-1]
                tag['high_watermark'] = max(tag['high_watermark'], current_high)
                trail_stop = tag['high_watermark'] - box_size
                emergency_stop = entry - 2 * box_size

                if self.data.Low[-