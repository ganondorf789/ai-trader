```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

class VoltaicReversal(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # Calculate all indicators using TA-Lib wrapped in self.I()
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR')
        self.atr_avg = self.I(talib.SMA, self.atr, timeperiod=14, name='ATR_Avg')
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=50, name='EMA50')
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14, name='RSI')
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='SwingHigh')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='SwingLow')

    def next(self):
        # Moon Dev safety check - ensure enough data points ✨
        if len(self.data) < 50:
            return

        current_close = self.data.Close[-1]
        current_atr = self.atr[-1]
        current_atr_avg = self.atr_avg[-1]

        # Moon Dev risk calculation 🌙
        risk_amount = self.equity * self.risk_percent

        if not self.position:
            # Long Entry Constellation 🌌
            long_conditions = (
                current_atr > 2 * current_atr_avg,
                current_close > self.ema50[-1],
                self.rsi[-1] < 30,
                self.data.Low[-1] <= self.swing_low[-1],
                current_close > self.swing_low[-1]
            )

            # Short Entry Constellation 🌌
            short_conditions = (
                current_atr > 2 * current_atr_avg,
                current_close < self.ema50[-1],
                self.rsi[-1] > 70,
                self.data.High[-1] >= self.swing_high[-1],
                current_close < self.swing_high[-1]
            )

            if all(long_conditions):
                sl_price = self.swing_low[-1] - 0.5 * current_atr
                risk_per_share = current_close - sl_price
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    tp_price = current_close + 1.5 * current_atr
                    print(f"🌙✨🚀 LONG SIGNAL! Buying {size} @ {current_close:.2f}")
                    self.buy(size=size, sl=sl_price, tp=tp_price)

            elif all(short_conditions):
                sl_price = self.swing_high[-1] + 0.5 * current_atr
                risk_per_share = sl_price - current_close
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    tp_price = current_close - 1.5 * current_atr
                    print(f"🌙✨🚀 SHORT SIGNAL! Selling {size} @ {current_close:.2f}")
                    self.sell(size=size, sl=sl_price, tp=tp_price)

        # Moon Dev trend reversal exits 🌗
        if self.position.is_long and crossunder(self.data.Close, self.ema50):
            print(f"🌙✨ Closing LONG @ {current_close:.2f} (EMA50 Reversal)")
            self.position.close()
        elif self.position.is_short and crossover(self.data.Close, self.ema50):
            print(f"🌙✨ Closing SHORT @ {current_close:.2f} (EMA50 Reversal)")
            self.position.close()

# Moon Dev data preparation ritual 🌙
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Cleanse the data columns 🔮
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime