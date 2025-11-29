Here's the fixed code with all the necessary changes and Moon Dev themed debug prints:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev Data Preparation 🛠️
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

# Cleanse and prepare cosmic data 🌌
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

class BandedMomentum(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌕
    
    def init(self):
        # 🌗 Celestial Indicators Configuration
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14, name='RSI')
        
        # Bollinger Bands using TA-Lib constellation 🌐
        upper, middle, lower = talib.BBANDS(self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        self.upper_bb = self.I(lambda: upper, name='UpperBB')
        self.middle_bb = self.I(lambda: middle, name='MiddleBB')
        self.lower_bb = self.I(lambda: lower, name='LowerBB')
        
        # Galactic trend filter 🌠
        self.sma50 = self.I(talib.SMA, self.data.Close, timeperiod=50, name='SMA50')
        
        # Cosmic support/resistance levels 🌑
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='SwingHigh')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='SwingLow')
        
        print("🌙 Moon Dev Indicators Activated! RSI(14) | BB(20,2) | SMA50 | Swing(20) ✨")

    def next(self):
        # Skip if stardust not settled 🌪️
        if len(self.data) < 50 or self.position:
            return

        # 🌓 Current planetary alignments
        c = {
            'rsi': self.rsi[-1],
            'prev_rsi': self.rsi[-2],
            'close': self.data.Close[-1],
            'prev_close': self.data.Close[-2],
            'upper_bb': self.upper_bb[-1],
            'middle_bb': self.middle_bb[-1],
            'lower_bb': self.lower_bb[-1],
            'sma50': self.sma50[-1],
            'swing_high': self.swing_high[-1],
            'swing_low': self.swing_low[-1]
        }

        # 🌕 Long Entry: RSI breaks Upper BB & Price breaks Middle BB
        long_trigger = (
            c['prev_rsi'] <= c['upper_bb'] and
            c['rsi'] > c['upper_bb'] and
            c['prev_close'] <= c['middle_bb'] and
            c['close'] > c['middle_bb'] and
            c['close'] > c['sma50']
        )

        # 🌑 Short Entry: RSI breaks Lower BB & Price breaks Middle BB
        short_trigger = (
            c['prev_rsi'] >= c['lower_bb'] and
            c['rsi'] < c['lower_bb'] and
            c['prev_close'] >= c['middle_bb'] and
            c['close'] < c['middle_bb'] and
            c['close'] < c['sma50']
        )

        # 🚀 Execute Lunar Trade Missions
        if long_trigger:
            self.execute_trade(direction='long', entry=self.data.Open[-1], sl=c['swing_low'])
            
        if short_trigger:
            self.execute_trade(direction='short', entry=self.data.Open[-1], sl=c['swing_high'])

    def execute_trade(self, direction, entry, sl):
        # 🌌 Calculate position size with cosmic precision
        risk = abs(entry - sl)
        if risk == 0:
            print("🌘