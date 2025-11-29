Here's the complete implementation of the HAVOC Momentum strategy for backtesting.py:

```python
# 🌙 Moon Dev's HAVOC Momentum Backtest 🌙
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
import numpy as np

# Clean and prepare data 🌌
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open', 'high': 'High',
    'low': 'Low', 'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class HavocMomentum(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌕
    max_daily_loss = 0.05  # 5% max daily loss 🚨
    sar_acceleration = 0.02
    sar_maximum = 0.2
    vwap_period = 20
    tsi_short = 14
    tsi_long = 25
    volume_sma_period = 20
    swing_period = 5  # Swing high/low lookback 🌙

    def init(self):
        # Heikin-Ashi Candles ✨
        ha_df = self.data.df[['Open','High','Low','Close']].ta.heikin_ashi()
        self.ha_open = self.I(lambda x: ha_df[f'HA_open'].values, name='HA Open')
        self.ha_close = self.I(lambda x: ha_df[f'HA_close'].values, name='HA Close')
        
        # True Strength Index 🌗
        self.tsi = self.I(ta.tsi, self.data.Close, self.tsi_long, self.tsi_short, name='TSI')
        
        # Volume-Weighted VWAP 🌊
        typical_price = (self.data.High + self.data.Low + self.data.Close) / 3
        vwap_num = self.I(talib.SUM, typical_price * self.data.Volume, timeperiod=self.vwap_period)
        vwap_den = self.I(talib.SUM, self.data.Volume, timeperiod=self.vwap_period)
        self.vwap = self.I(lambda: vwap_num / vwap_den, name='VWAP')
        
        # Parabolic SAR 🎯
        self.sar = self.I(talib.SAR, self.data.High, self.data.Low, 
                         acceleration=self.sar_acceleration, 
                         maximum=self.sar_maximum, name='SAR')
        
        # Volume & Swing Levels 🌑
        self.volume_sma = self.I(talib.SMA, self.data.Volume, self.volume_sma_period)
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period)
        
        # Track daily equity 🌓
        self.daily_equity = [self.equity]

    def next(self):
        # 🌙 Moon Dev Debug Prints 🌙
        print(f"\n🌌 Processing {self.data.index[-1]} | Close: {self.data.Close[-1]:.2f}")
        
        # Check trading hours ⏳
        current_time = self.data.index[-1].time()
        if (current_time.hour == 0 and current_time.minute < 30) or \
           (current_time.hour == 23 and current_time.minute >= 30):
            print(f"🌙 Skipping first/last 30m of session")
            return

        # Risk Management Checks 🛡️
        if self.equity < self.daily_equity[-1] * (1 - self.max_daily_loss):
            print(f"🚨 EMERGENCY: Max daily loss hit! Closing all positions.")
            self.position.close()
            return
        
        # Update daily equity tracker 🌓
        if len(self.data) % 96 == 0:  # 24h in 15m intervals
            self.daily_equity.append(self.equity)

        # Entry/Exit Logic 🚀
        if self.position:
            # Parabolic SAR Exit 🎯
            if (self.position.is