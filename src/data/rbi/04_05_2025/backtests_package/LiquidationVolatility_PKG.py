I'll fix the code by removing the `backtesting.lib` import and ensuring all indicators use proper packages. Here's the corrected version with Moon Dev themed debug prints:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import numpy as np  # Added for numerical operations

# Data preprocessing
data = pd.read_csv(
    '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
    parse_dates=['datetime'],
    index_col='datetime'
)

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class LiquidationVolatility(Strategy):
    risk_pct = 0.01
    atr_period = 4        # 1-hour ATR (4*15min)
    atr_ma_period = 1920  # 20-day SMA (1920*15min)
    swing_period = 20
    lookback = 200
    trail_multiplier = 2
    
    def init(self):
        # Volatility indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=self.atr_ma_period)
        
        # Liquidation zones
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period)
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.swing_period)
        
        # Liquidation signals
        self.liq_high = self.I(
            lambda: (self.data.High == self.swing_high) & (self.data.Volume > 2*self.vol_ma),
            name='LIQUIDATION_HIGH'
        )
        self.liq_low = self.I(
            lambda: (self.data.Low == self.swing_low) & (self.data.Volume > 2*self.vol_ma),
            name='LIQUIDATION_LOW'
        )
        
        print("🌕 Moon Dev Indicators Initialized! Lunar trading cycles activated.")
        
    def next(self):
        # Moon Dev progress tracker
        if len(self.data) % 500 == 0:
            print(f"🌙 Lunar Cycle Update: Processing candle {len(self.data)} - Current Phase: {np.round(self.equity/10000, 2)} moons")
            
        if len(self.data) < self.atr_ma_period or self.position:
            return
            
        current_atr = self.atr[-1]
        current_atr_ma = self.atr_ma[-1]
        
        if current_atr > current_atr_ma:
            # Find recent liquidation levels
            start = max(0, len(self.data)-self.lookback)
            resistance_levels = [self.data.High[i] 
                               for i in range(start, len(self.data)-1) 
                               if self.liq_high[i]]
            support_levels = [self.data.Low[i] 
                            for i in range(start, len(self.data)-1) 
                            if self.liq_low[i]]
            
            # Check for breakout signals
            current_close = self.data.Close[-1]
            proximity = current_atr * 0.5
            
            if resistance_levels:
                resistance = max(resistance_levels)
                if (current_close > resistance and
                    abs(current_close - resistance) <= proximity):
                    self.enter_trade('long', resistance, current_atr)
                    print(f"🚀 Moon Launch Detected! Long signal at {current_close}")
                    
            if support_levels:
                support = min(support_levels)
                if (current_close < support and
                    abs(current_close - support) <= proximity):
                    self.enter_trade('short', support, current_atr)
                    print(f"🌑 Black Moon Alert! Short signal at {current_close}")
                    
        # Manage existing positions
        for trade in self.trades:
            if trade.is_long:
                self.manage_long(trade)
            else:
                self.manage_short(trade)
                
    def enter_trade(self, direction, level,