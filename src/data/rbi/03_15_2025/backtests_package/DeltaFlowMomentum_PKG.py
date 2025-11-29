I'll help fix the code by removing the `backtesting.lib` import and ensuring all indicators are properly implemented. Here's the corrected version:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 MOON DEV DATA PREPARATION 🌙
def prepare_data(path):
    data = pd.read_csv(path)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Drop unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    # Proper column mapping
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    # Convert to datetime index
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

# 🌟 DELTAFLOW MOMENTUM STRATEGY 🌟
class DeltaFlowMomentum(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    sma_period = 20
    delta_window = 14
    spread_ma_period = 5
    stop_pct = 0.01
    
    def init(self):
        # 🚀 CORE INDICATORS
        # Spread analysis
        self.spread = self.I(lambda x: x.High - x.Low, self.data)
        self.spread_ma = self.I(talib.SMA, self.spread, timeperiod=self.spread_ma_period)
        
        # Delta calculations
        close_open_diff = self.I(lambda x: x.Close - x.Open, self.data)
        self.delta = self.I(lambda x: (x.Close - x.Open) * x.Volume, self.data)
        self.cum_delta = self.I(talib.SUM, self.delta, timeperiod=self.delta_window)
        
        # Momentum confirmation
        self.sma20 = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_period)
        
        print("🌙 DeltaFlow Momentum Strategy Activated! 🚀✨")

    def next(self):
        # 📊 CHECK INDICATOR READINESS
        if len(self.data.Close) < self.delta_window + 2:
            return

        # 🎯 CURRENT VALUES
        price = self.data.Close[-1]
        spread_ma = self.spread_ma[-1]
        prev_spread_ma = self.spread_ma[-2]
        cum_delta = self.cum_delta[-1]
        prev_cum_delta = self.cum_delta[-2]
        sma20 = self.sma20[-1]

        # 🌙 MOON DEV SIGNAL DETECTION
        long_signal = (
            spread_ma < prev_spread_ma and  # Spread narrowing
            cum_delta > prev_cum_delta and  # Delta accelerating
            cum_delta > 0 and              # Positive delta
            price > sma20                  # Momentum confirmation
        )

        short_signal = (
            spread_ma > prev_spread_ma and  # Spread widening
            cum_delta < prev_cum_delta and  # Delta decelerating
            cum_delta < 0 and              # Negative delta
            price < sma20                  # Momentum confirmation
        )

        # 💼 RISK MANAGEMENT CALCULATIONS
        if not self.position:
            if long_signal:
                sl_price = price * (1 - self.stop_pct)
                risk_amount = self.equity * self.risk_pct
                risk_per_share = price - sl_price
                size = int(round(risk_amount / risk_per_share))
                if size > 0:
                    self.buy(size=size, sl=sl_price)
                    print(f"🌙 LONG ENTRY! Size: {size} ✨ Price: {price:.2f} 🚀")
            
            elif short_signal:
                sl_price = price * (1 + self.stop_pct)
                risk_amount = self.equity * self.risk_pct
                risk_per_share = sl_price - price
                size = int(round(risk_amount / risk_per_share))
                if size > 0:
                    self.sell(size=size, sl=sl_price)
                    print(f"🌙 SHORT ENTRY! Size: {size} ✨ Price: {price:.2f} 🚀")

        else:
            # 🛑 EXIT CONDITIONS
            if self.position.is_long:
                if (spread