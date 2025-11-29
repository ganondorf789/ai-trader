Here's the fixed code with all backtesting.lib references removed and Moon Dev themed improvements:

```python
# 🌙 Moon Dev's Liquidation Fade Backtest 🌙
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
import numpy as np

# Clean and prepare cosmic data 🌌
def prepare_data(path):
    data = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    
    # Cosmic column alignment ✨
    column_map = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    data.rename(columns=column_map, inplace=True)
    return data

class LiquidationFade(Strategy):
    risk_pct = 0.01  # 1% cosmic dust risk per trade 🌠
    bb_period = 20
    bb_dev = 2
    keltner_mult = 1.5
    liq_lookback = 672  # 7 days in 15m intervals
    stop_pct = 0.02
    
    def init(self):
        # 🌗 Bollinger Bands (Upper Band Focus)
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, 
                              self.bb_period, self.bb_dev, self.bb_dev, 0,
                              name='BB_UP', position=0)
        
        # 🌍 Keltner Channel Calculations
        typical_price = (self.data.High + self.data.Low + self.data.Close) / 3
        self.keltner_mid = self.I(talib.EMA, typical_price, self.bb_period,
                                name='KELTNER_MID')
        atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                    self.bb_period, name='ATR')
        
        # 🌊 Liquidation Force Field
        self.liq_median = self.I(ta.median, self.data['liquidations'], 
                                self.liq_lookback, name='LIQ_MEDIAN')
        
        # ⚡ Volatility Filter
        bb_width = self.I(lambda u, l, m: (u - l)/m, 
                          self.bb_upper, 
                          self.I(talib.BBANDS, self.data.Close, self.bb_period, 
                                self.bb_dev, self.bb_dev, 0, position=2),
                          self.I(talib.BBANDS, self.data.Close, self.bb_period, 
                                self.bb_dev, self.bb_dev, 0, position=1),
                          name='BB_WIDTH')
        self.bb_width_ma = self.I(talib.SMA, bb_width, 20, name='BB_WIDTH_MA')
        
    def next(self):
        # 🌑 Moon-powered Entry Logic
        if not self.position:
            liq_spike = self.data['liquidations'][-1] > 2 * self.liq_median[-1]
            price_breach = self.data.Close[-1] > self.bb_upper[-1]
            
            if liq_spike and price_breach:
                # 🚀 Position Sizing with Cosmic Precision
                equity = self.equity()
                entry_price = self.data.Close[-1]
                stop_price = entry_price * (1 + self.stop_pct)
                risk_per_share = stop_price - entry_price
                position_size = (equity * self.risk_pct) / risk_per_share
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.sell(size=position_size, sl=stop_price, 
                             tag="🌕 SHORT ENTRY")
                    print(f"🚀🌑 MOON DEV ALERT: Shorting {position_size} units at {entry_price}! Liquidations spiking 🌊")

        # 🌕 Exit When Volatility Contracts
        else:
            # Replaced crossover with direct comparison
            mid_cross = (self.keltner_mid[-2] < self.data.Close[-2] and 
                         self.keltner_mid[-1] > self.data.Close[-1])
            volatility_ok = self.bb_width_ma[-1] > self.bb_width_ma[-1]
            
            if mid_cross and volatility_ok:
                self.position.close()
                print(f"🌖 MOON DEV EXIT: Closing at {