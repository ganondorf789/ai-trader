Here's the fixed and completed backtest code with proper Moon Dev debugging improvements 🌙:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from collections import defaultdict

# Load and preprocess data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class LiquidationDivergence(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    max_daily_trades = 3
    rsi_period = 14
    bb_period = 20
    bb_dev = 2
    liquidation_window = 2880  # 30 days in 15m intervals
    swing_period = 20

    def init(self):
        # Moon Dev Indicator Setup 🌙
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period, name='RSI')
        
        # Bollinger Bands Setup
        self.upper_bb, self.middle_bb, self.lower_bb = self.I(
            talib.BBANDS, self.data.Close, self.bb_period, self.bb_dev, self.bb_dev, name='BBANDS'
        )

        # Liquidation Analysis 🌊
        self.liquidation_ma = self.I(talib.SMA, self.data['liquidation'], self.liquidation_window, name='Liq MA')
        self.liquidation_std = self.I(talib.STDDEV, self.data['liquidation'], self.liquidation_window, name='Liq STD')
        self.liq_upper = self.I(lambda x, y: x + y*2, self.liquidation_ma, self.liquidation_std, name='2σ Upper')
        self.liq_lower = self.I(lambda x, y: x - y*2, self.liquidation_ma, self.liquidation_std, name='2σ Lower')

        # Swing Points ✨
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period, name='Swing High')
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period, name='Swing Low')

        # Trade Counter 📆
        self.daily_trades = defaultdict(int)

    def next(self):
        current_date = self.data.index[-1].date()
        if self.daily_trades[current_date] >= self.max_daily_trades:
            print("🌑 MOON ALERT: Max daily trades reached - no more trades today")
            return  # 🚫 Max trades reached for today

        # Moon Dev Signal Calculations 🌙
        liq = self.data['liquidation'][-1]
        liq_signal = 'NEUTRAL'
        if liq > self.liq_upper[-1]: 
            liq_signal = 'SHORT'
            print("🌘 MOON SIGNAL: Liquidation spike detected (SHORT bias)")
        elif liq < self.liq_lower[-1]: 
            liq_signal = 'LONG'
            print("🌖 MOON SIGNAL: Liquidation dip detected (LONG bias)")

        # RSI Divergence Detection 🔄
        bullish_div = bearish_div = False
        if len(self.rsi) > 3:
            bullish_div = (self.data.Low[-1] < self.data.Low[-2] and 
                          self.rsi[-1] > self.rsi[-2])
            bearish_div = (self.data.High[-1] > self.data.High[-2] and 
                          self.rsi[-1] < self.rsi[-2])

        # Moon Dev Entry Logic 🚀
        if not self.position:
            if liq_signal == 'SHORT' and bearish_div:
                sl_price = self.swing_high[-1] * 1.005  # 0.5% above swing high
                risk_amount = self.equity * self.risk_pct
                position_size = risk_amount / (self.data.Close[-1] - sl_price)
                print(f"🌒