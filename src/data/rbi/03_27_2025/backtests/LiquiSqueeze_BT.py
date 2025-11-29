```python
# 🌙 Moon Dev's LiquiSqueeze Backtest 🌙
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Data Preparation 🌍
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class LiquiSqueeze(Strategy):
    # Strategy Parameters 🚀
    bb_period = 20
    bb_dev = 2
    bbw_threshold = 0.5
    swing_period = 20
    base_risk_pct = 0.01
    max_risk_pct = 0.02
    tp_multiplier = 1.5
    partial_close_ratio = 0.5
    liquidity_proximity = 0.02  # 2%

    def init(self):
        # Core Indicators Calculation 🌗
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(talib.BBANDS, 
            self.data.Close, 
            timeperiod=self.bb_period,
            nbdevup=self.bb_dev,
            nbdevdn=self.bb_dev,
            matype=0
        )
        
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period)
        self.volume_ma = self.I(talib.SMA, self.data.Volume, 20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)

    def next(self):
        # Current Market Conditions 🌡️
        price = self.data.Close[-1]
        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]
        bbw = (bb_upper - bb_lower) / self.bb_middle[-1]
        
        if not self.position:
            # Long Entry Logic 🌕
            long_trigger = (price > bb_upper) or (price > self.swing_high[-1])
            if (long_trigger and 
                bbw < self.bbw_threshold and
                self.data.Volume[-1] > self.volume_ma[-1]):
                
                # Risk Management Calculations 🧮
                stop_loss = bb_lower
                risk_pct = self.max_risk_pct if self._near_liquidation(price, self.swing_high[-1]) else self.base_risk_pct
                risk_amount = self.equity * risk_pct
                risk_distance = price - stop_loss
                
                if risk_distance > 0:
                    position_size = int(round(risk_amount / risk_distance))
                    self.buy(
                        size=position_size, 
                        sl=stop_loss,
                        tag={
                            'squeeze_range': bb_upper - bb_lower,
                            'entry_type': 'long'
                        }
                    )
                    print(f"🌙 LONG ENTRY @ {price:.2f} | Size: {position_size} ✨")

            # Short Entry Logic 🌑
            short_trigger = (price < bb_lower) or (price < self.swing_low[-1])
            if (short_trigger and 
                bbw < self.bbw_threshold and
                self.data.Volume[-1] > self.volume_ma[-1]):
                
                stop_loss = bb_upper
                risk_pct = self.max_risk_pct if self._near_liquidation(price, self.swing_low[-1]) else self.base_risk_pct
                risk_amount = self.equity * risk_pct
                risk_distance = stop_loss - price
                
                if risk_distance > 0:
                    position_size = int(round(risk_amount / risk_distance))
                    self.sell(
                        size=position_size, 
                        sl=stop_loss,
                        tag={
                            'squeeze_range': bb_upper - bb_lower,
                            'entry_type': 'short'
                        }
                    )
                    print(f"