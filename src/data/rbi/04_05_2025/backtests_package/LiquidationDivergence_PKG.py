Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations:

```python
# 🌙 Moon Dev's Liquidation Divergence Backtest 🌙
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

# 🚀 Data Preparation Magic 🚀
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
}, inplace=True)

class LiquidationDivergence(Strategy):
    # 🌙 Strategy Parameters
    risk_pct = 0.02
    rsi_period = 14
    liquidation_window = 20
    atr_period = 14
    exit_after_bars = 5

    def init(self):
        # 🌟 Core Indicators
        self.lq_high = self.I(talib.MAX, self.data.High, timeperiod=self.liquidation_window, name='LIQUIDATION HIGH 🌊')
        self.lq_low = self.I(talib.MIN, self.data.Low, timeperiod=self.liquidation_window, name='LIQUIDATION LOW 🌊')
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period, name='RSI 📉')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR ⚡')
        
        # 🌐 Exchange Flow Simulation
        self.flow_short = self.I(talib.SMA, self.data.Volume, 5, name='FLOW SHORT 🌌')
        self.flow_long = self.I(talib.SMA, self.data.Volume, 20, name='FLOW LONG 🌠')
        self.net_flow = self.flow_short - self.flow_long
        
        # 📊 Volume Profile
        self.vol_profile = self.I(talib.SMA, self.data.Close * self.data.Volume, 50, name='VOL PROFILE 🔊')

    def next(self):
        # 🌙 Avoid Early Calculations
        if len(self.data) < 50:
            return

        # 🎯 Current Values
        price = self.data.Close[-1]
        lq_high = self.lq_high[-1]
        lq_low = self.lq_low[-1]
        rsi = self.rsi[-1]
        atr = self.atr[-1]
        flow = self.net_flow[-1]

        # 🌓 Moon Dev Entry Conditions
        long_signal = (
            (price <= lq_low * 1.01) and
            (flow > self.net_flow[-3]) and
            (rsi < 35) and
            (self.data.Volume[-1] > self.vol_profile[-1])
        )

        short_signal = (
            (price >= lq_high * 0.99) and
            (flow < self.net_flow[-3]) and
            (rsi > 65) and
            (self.data.Volume[-1] > self.vol_profile[-1])
        )

        # 🌕 Position Sizing Alchemy
        equity = self.equity
        risk_amount = equity * self.risk_pct

        # 🚀 Execution Logic
        if not self.position:
            if long_signal:
                sl = lq_low - 1.7 * atr
                tp = lq_high + 1.7 * atr
                risk_per_unit = price - sl
                if risk_per_unit > 0:
                    size = int(round(risk_amount / risk_per_unit))
                    self.buy(size=size, sl=sl, tp=tp)
                    print(f"🌕 MOON DEV LONG! 🌙 | Size: {size} | Entry: {price:.2f}")
                    print(f"🔮 SL: {sl:.2f} | 🌠 TP: {tp:.2f}")

            elif short_signal:
                sl = lq_high + 1.7 * atr
                tp = lq_low - 1.7 * atr
                risk_per_unit = sl - price
                if risk_per_unit > 0:
                    size = int(round(risk_amount