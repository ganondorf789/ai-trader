Here's the fixed code with all backtesting.lib imports removed and crossover functions replaced with proper implementations:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

class VolatilityFilteredEMA(Strategy):
    ema_period_short = 50
    ema_period_long = 200
    hv_lookback = 20
    percentile_window = 100
    risk_pct = 0.01  # 1% risk per trade
    atr_period = 14
    atr_multiplier = 2

    def init(self):
        # Core trend indicators 🌙
        self.ema_short = self.I(talib.EMA, self.data.Close, self.ema_period_short, name='EMA50')
        self.ema_long = self.I(talib.EMA, self.data.Close, self.ema_period_long, name='EMA200')
        
        # Volatility calculations ✨
        price_changes = self.I(talib.MOM, self.data.Close, 1, name='PriceChanges')
        self.hv = self.I(talib.STDDEV, price_changes, self.hv_lookback, name='HV')
        self.hv_low = self.I(ta.quantile, self.hv, self.percentile_window, q=0.2, name='HV20pct')
        self.hv_high = self.I(ta.quantile, self.hv, self.percentile_window, q=0.8, name='HV80pct')
        
        # Risk management indicators 🛡️
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR')

    def next(self):
        # Wait for full indicator warmup 🌡️
        if len(self.data) < self.percentile_window:
            return

        current_close = self.data.Close[-1]
        trade_size = None

        # Entry logic: Golden Cross + Low Volatility 🌙🚀
        if not self.position:
            # Bullish crossover check (replaced backtesting.lib.crossover)
            golden_cross = (self.ema_short[-2] < self.ema_long[-2] and 
                          self.ema_short[-1] > self.ema_long[-1])
            
            if golden_cross and self.hv[-1] < self.hv_low[-1]:
                # ATR-based position sizing 📈
                risk_amount = self.broker.equity * self.risk_pct
                atr_value = self.atr[-1] or 0.0001  # Avoid zero division
                stop_loss = self.atr_multiplier * atr_value
                trade_size = risk_amount / stop_loss
                trade_size = int(round(trade_size))

                if trade_size > 0:
                    self.buy(size=trade_size, sl=current_close-stop_loss)
                    print(f"🌙✨🚀 MOON DEV ALERT: Long Entry @ {current_close:.2f} | Size: {trade_size} | SL: {current_close-stop_loss:.2f}")

        # Exit logic: Death Cross or High Volatility 🌙💥
        else:
            # Bearish crossover check (replaced backtesting.lib.crossover)
            death_cross = (self.ema_long[-2] < self.ema_short[-2] and 
                         self.ema_long[-1] > self.ema_short[-1])
            
            exit_condition = (death_cross or self.hv[-1] > self.hv_high[-1])
            
            if exit_condition:
                self.position.close()
                reason = "Death Cross" if death_cross else "Volatility Spike"
                print(f"🌙🌑 MOON DEV EXIT: {reason} @ {current_close:.2f}")

# Data preparation 🌙📊
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean and format data 🧹✨
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data.set_index(pd.to_datetime(data['datetime']), inplace=True)

# Execute back