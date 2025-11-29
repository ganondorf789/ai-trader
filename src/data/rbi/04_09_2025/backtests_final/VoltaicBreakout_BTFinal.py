Here's the fully debugged and optimized code with Moon Dev enhancements while preserving the original strategy logic:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import numpy as np
from backtesting import Strategy, Backtest

class VoltaicBreakout(Strategy):
    bb_period = 20
    bb_dev = 2
    adx_period = 14
    risk_pct = 0.01
    max_positions = 3
    
    def init(self):
        # Moon Dev Indicator Initialization 🌙
        print("🌙 Initializing Moon Dev Trading System... ✨")
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, self.adx_period)
        self.middle = self.I(talib.SMA, self.data.Close, timeperiod=self.bb_period)
        self.stddev = self.I(talib.STDDEV, self.data.Close, timeperiod=self.bb_period, nbdev=1)
        self.upper = self.I(lambda: self.middle.array + 2*self.stddev.array)
        self.lower = self.I(lambda: self.middle.array - 2*self.stddev.array)
        
        # Moon Dev Trade Counter 🌙
        self.trade_counter = 0
        print("🌙 Indicators successfully initialized! ✨")

    def next(self):
        current_close = self.data.Close[-1]
        current_adx = self.adx[-1]
        prev_adx = self.adx[-2] if len(self.adx) > 1 else 0

        # Moon Dev Trend Filter 🌙
        adx_cross_above_25 = current_adx > 25 and prev_adx <= 25
        upper_band = self.upper[-1]
        lower_band = self.lower[-1]

        # Moon Dev Exit Condition ✨
        if current_adx < 20 and self.trades:
            for trade in list(self.trades):
                trade.close()
            print(f"🌙 Moon Dev Trend Exit! ADX {current_adx:.1f} < 20 ✨")

        # Moon Dev Entry Logic 🚀
        if not self.position and len(self.trades) < self.max_positions:
            if current_close > upper_band and adx_cross_above_25:
                sl_price = self.lower[-1]
                risk_per_share = current_close - sl_price
                
                if risk_per_share > 0:
                    risk_amount = self.equity * self.risk_pct
                    position_size = int(round(risk_amount / risk_per_share))
                    tp_price = current_close + 2 * risk_per_share
                    
                    if position_size > 0:
                        self.trade_counter += 1
                        self.buy(size=position_size, sl=sl_price, tp=tp_price, tag=f"LONG-{self.trade_counter}")
                        print(f"🌙 Moon Dev LONG! 🚀 Entry: {current_close:.2f}, Size: {position_size}")
                        print(f"   SL: {sl_price:.2f}, TP: {tp_price:.2f} ✨")

            elif current_close < lower_band and adx_cross_above_25:
                sl_price = self.upper[-1]
                risk_per_share = sl_price - current_close
                
                if risk_per_share > 0:
                    risk_amount = self.equity * self.risk_pct
                    position_size = int(round(risk_amount / risk_per_share))
                    tp_price = current_close - 2 * risk_per_share
                    
                    if position_size > 0:
                        self.trade_counter += 1
                        self.sell(size=position_size, sl=sl_price, tp=tp_price, tag=f"SHORT-{self.trade_counter}")
                        print(f"🌙 Moon Dev SHORT! 🚀 Entry: {current_close:.2f}, Size: {position_size}")
                        print(f"   SL: {sl_price:.2f}, TP: {tp_price:.2f} ✨")

# Moon Dev Data Preparation 🌙
print("🌙 Loading celestial market data... ✨")
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
                   parse_dates=['datetime'], index_col='datetime')
data.columns = [col.strip().lower() for col in data.columns]
data = data.loc[:, ~data.columns.str.contains('^unnamed')]

# Moon Dev Backtest Setup 🌙
print("🌙 Preparing lunar backtest environment... ✨")
bt = Backtest(data, Voltaic