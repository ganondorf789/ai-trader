I'll complete and debug the VolatilityAnchorVwap strategy while maintaining all the original logic and adding Moon Dev themed debug prints. Here's the fixed version:

```python
import pandas as pd
import talib
from backtesting import Strategy, Backtest
import numpy as np

# Data preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and prepare data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolatilityAnchorVwap(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    max_positions = 3
    daily_loss_limit = 0.05  # 5% daily loss limit
    weekly_dd_limit = 0.10  # 10% weekly drawdown limit
    
    def init(self):
        # Volatility indicators
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, 20, 2, 2)[0]
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, 20, 2, 2)[2]
        self.bb_mid = self.I(talib.SMA, self.data.Close, 20)
        self.bb_width = (self.bb_upper - self.bb_lower)/self.bb_mid
        self.bb_width_pct = self.I(talib.MAX, self.bb_width, 288)  # 3-day lookback
        
        # Volume indicators
        self.vol_ma = self.I(talib.SMA, self.data.Volume, 20*96)  # 20-day MA
        
        # Momentum indicators
        self.rsi = self.I(talib.RSI, self.data.Close, 14)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
        # VWAP anchors
        self.cum_typical_vol = ((self.data.High + self.data.Low + self.data.Close)/3 * 
                               self.data.Volume).cumsum().values
        self.cum_vol = self.data.Volume.cumsum().values
        self.active_anchors = []
        
        # Risk management tracking
        self.current_day = self.data.index[0].date()
        self.day_start_equity = self._broker.starting_cash
        self.week_start_equity = self._broker.starting_cash

    def next(self):
        # Moon-themed debug prints 🌙✨
        if len(self.data) % 100 == 0:
            print(f"🌙 Processing bar {len(self.data)} - Moon Dev tracking progress...")
        
        # Risk management checks
        self._check_daily_loss()
        self._check_weekly_dd()
        
        # Detect new volume spikes
        if self.data.Volume[-1] > 3 * self.vol_ma[-1]:
            print(f"🚀 Volume spike detected! {self.data.Volume[-1]:.2f} vs {self.vol_ma[-1]:.2f} - Moon Dev alert!")
            anchor_idx = len(self.data)-1
            cum_typ_start = self.cum_typical_vol[anchor_idx-1] if anchor_idx > 0 else 0
            cum_vol_start = self.cum_vol[anchor_idx-1] if anchor_idx > 0 else 0
            self.active_anchors.append({
                'idx': anchor_idx,
                'cum_typ': cum_typ_start,
                'cum_vol': cum_vol_start,
                'age': 0
            })
        
        # Update anchors and check entries
        for anchor in list(self.active_anchors):
            anchor['age'] += 1
            if anchor['age'] > 5:
                self.active_anchors.remove(anchor)
                print(f"🌑 Anchor expired after 5 periods - Moon Dev clearing old signals")
                continue
                
            current_cum_typ = self.cum_typical_vol[-1]
            current_cum_vol = self.cum_vol[-1]
            vwap = (current_cum_typ - anchor['cum_typ'])/(current_cum_