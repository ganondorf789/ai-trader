Here's the fixed and complete code with Moon Dev themed improvements while maintaining the original strategy logic:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import numpy as np
from backtesting import Strategy, Backtest

# Clean and prepare data
def prepare_data(filepath):
    # Load data with proper cleaning
    data = pd.read_csv(filepath)
    
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    
    # Drop unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    
    # Map columns to required format
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
    # Ensure proper datetime format
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    
    return data

# Moon Dev's Liquidity Bandit Strategy 🌙💎
class LiquidityBandit(Strategy):
    # Strategy parameters
    bb_period = 20
    vol_ma_period = 20
    liquidity_roc_threshold = 0.5
    risk_per_trade = 0.01
    max_hold_bars = 5
    
    def init(self):
        # Calculate Bollinger Bands
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, timeperiod=self.bb_period,
                              nbdevup=2, nbdevdn=2, matype=0, name='BB_UPPER', which=0)
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=self.bb_period,
                              nbdevup=2, nbdevdn=2, matype=0, name='BB_LOWER', which=2)
        
        # Calculate Bollinger Band Width (Moon Dev's special formula ✨)
        self.bb_width = self.I(lambda u, l: (u - l)/self.bb_upper,
                               self.bb_upper, self.bb_lower, name='BB_WIDTH')
        self.bb_width_avg = self.I(talib.SMA, self.bb_width, timeperiod=self.bb_period,
                                  name='BB_WIDTH_AVG')
        
        # Volume indicators (Lunar volume scanner engaged 🌕)
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_ma_period,
                            name='VOLUME_MA')
        
        # Liquidity indicators (using exchange net flows)
        self.liquidity_roc = self.I(talib.ROC, self.data['Exchange_Net_Flows'], 
                                   timeperiod=2, name='LIQUIDITY_ROC')
        
        print("🌙✨ Moon Dev Indicators Activated! Lunar scanners engaged! 🛰️💫")
        print("🚀 All systems nominal - Ready for lunar trading operations! 🌌")

    def next(self):
        # Skip initial bars for indicator warmup
        if len(self.data) < self.bb_period * 2:
            return
        
        current_close = self.data.Close[-1]
        current_bb_width = self.bb_width[-1]
        bb_width_avg = self.bb_width_avg[-1]
        current_vol = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]
        liquidity_roc = self.liquidity_roc[-1]
        
        # Moon Dev's Entry Conditions 🌙
        entry_conditions = (
            current_bb_width < 0.2 * bb_width_avg and  # Bandwidth contraction
            current_vol > 2 * vol_ma and              # Volume spike
            abs(liquidity_roc) > self.liquidity_roc_threshold  # Liquidity shift
        )
        
        if entry_conditions and not self.position:
            # Determine trade direction based on liquidity flow
            if liquidity_roc > 0:
                # Long entry logic (Moon Shot! 🚀)
                entry_price = current_close
                sl_price = self.data.Low[-1] * 0.99  # 1% below breakout low
                risk_per_share = entry_price - sl_price
                
                if risk_per_share <= 0:
                    print("🌑 Warning: Invalid risk calculation - Aborting lunar launch!")
                    return
                
                # Position sizing (Moon Math engaged 🌙)
                risk_amount = self.risk_per_trade * self.equity
                position_size = int(round(risk_amount / risk_per_share))