Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np
from backtesting import Backtest, Strategy

def load_and_preprocess_data(data_path):
    # Load and clean data
    data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }).dropna()
    
    # Resample and calculate Heikin-Ashi for 4H timeframe
    resampled_4h = data.resample('4H').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'})
    ha_4h = resampled_4h.ta.heikinashi().rename(columns=lambda x: f"HA_{x}")
    data = data.join(ha_4h[['HA_close','HA_open']]).ffill()
    
    # Resample and calculate CMF for 1H timeframe
    resampled_1h = data.resample('1H').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'})
    resampled_1h['CMF'] = resampled_1h.ta.cmf(length=20)
    data = data.join(resampled_1h[['CMF']]).ffill()
    
    return data

class FractalDivergence(Strategy):
    risk_pct = 0.01
    max_concurrent_trades = 3
    atr_period = 14
    
    def init(self):
        # Multi-timeframe indicators
        self.ha_close = self.I(lambda x: x, self.data.HA_close, name='HA_Close')
        self.ha_open = self.I(lambda x: x, self.data.HA_open, name='HA_Open')
        self.cmf = self.I(lambda x: x, self.data.CMF, name='CMF')
        
        # Fractal calculations
        self.swing_highs = self.I(talib.MAX, self.data.High, 5, name='Swing High')
        self.swing_lows = self.I(talib.MIN, self.data.Low, 5, name='Swing Low')
        
        # Volatility measure
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)

    def next(self):
        # Moon Dev risk management check
        if len(self.trades) >= self.max_concurrent_trades:
            print("🌙 Moon Dev Alert: Max concurrent trades reached! 🌙")
            return

        # Get recent values for pattern detection
        current_close = self.data.Close[-1]
        ha_bullish = self.ha_close[-1] > self.ha_open[-1]
        ha_bearish = self.ha_close[-1] < self.ha_open[-1]
        
        # Fractal-based stop levels
        recent_swing_high = max(self.swing_highs[-10:])
        recent_swing_low = min(self.swing_lows[-10:])
        
        # Divergence detection window
        lookback = 8  # Approximately 2 hours in 15m intervals
        
        # Moon Dev themed debug prints
        if len(self.data) % 100 == 0:
            print(f"🌙 Moon Dev Status Update 🌙 | Equity: {self.equity:,.2f} | Price: {current_close:.2f} | Active Trades: {len(self.trades)}")
        
        # Long entry logic
        if not self.position and ha_bullish:
            # Check for bullish divergence
            price_lows = self.data.Low[-lookback:]
            cmf_values = self.cmf[-lookback:]
            
            lower_low = price_lows[-1] < price_lows[-lookback//2]
            higher_cmf = cmf_values[-1] > cmf_values[-lookback//2]
            
            if lower_low and higher_cmf:
                # Calculate position size
                sl = recent_swing_low
                risk_per_share = self.data.Close[-1] - sl
                
                if risk_per_share <= 0:
                    print("🌙 Moon Dev Warning: Invalid stop loss level!