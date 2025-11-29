```python
# 🌙 Moon Dev's LiquidationCluster Backtest 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import numpy as np

# Data preparation moon magic ✨
def load_data(path):
    data = pd.read_csv(
        path,
        parse_dates=['datetime'],
        index_col='datetime',
        dtype={'Volume': 'float'}
    )
    
    # Cleanse column names 🌌
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[
        col for col in data.columns 
        if 'unnamed' in col.lower()
    ])
    
    # Validate critical columns exist 🌕
    required_cols = {'open', 'high', 'low', 'close', 'volume', 'funding_rate'}
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"🚨 Missing columns: {missing}")
    
    # Proper case mapping for backtesting.py 🌗
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'funding_rate': 'Funding_Rate'
    })
    return data

class LiquidationCluster(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌓
    tp_multiplier = 2  # 2:1 reward:risk ratio 🌛
    
    def init(self):
        # Weekly VWAP Calculation 🌗
        close = self.data.Close
        volume = self.data.Volume
        self.sum_pv = self.I(talib.SUM, close * volume, 672, name='SUM_PV')
        self.sum_vol = self.I(talib.SUM, volume, 672, name='SUM_VOL')
        self.weekly_vwap = self.I(
            lambda x: x[0]/x[1] if x[1] != 0 else np.nan, 
            self.sum_pv, self.sum_vol, 
            name='Weekly_VWAP'
        )
        
        # Funding Rate Indicators 🌑
        self.funding_7d_avg = self.I(
            talib.SMA, self.data.Funding_Rate, 672, 
            name='Funding_7D_Avg'
        )
        
        # Swing High for Stop Loss 🌓
        self.swing_high = self.I(
            talib.MAX, self.data.High, 20, 
            name='20Period_Swing_High'
        )
        
        print("🌙✨ Strategy initialized with Moon Power! ✨")

    def next(self):
        current_dt = self.data.index[-1]
        current_price = self.data.Close[-1]
        
        # Skip weekend candles 🌈
        if current_dt.weekday() >= 5:
            print(f"🌙🌴 Weekend detected: {current_dt.date()} - Skipping trade")
            return
        
        # Moon Dev Debug Prints 🌕
        print(f"🌙 Processing {current_dt} | Price: {current_price:.2f}")
        
        # Manage existing position 🌘
        if self.position:
            # Check funding rate normalization exit
            if self.data.Funding_Rate[-1] < self.funding_7d_avg[-1]:
                print(f"🚨🌙 Funding rate normalized! Closing position")
                self.position.close()
            return
        
        # Entry Logic 🌗
        if len(self.data) < 2:
            return
            
        prev_close = self.data.Close[-2]
        prev_vwap = self.weekly_vwap[-2]
        prev_funding = self.data.Funding_Rate[-2]
        prev_avg_funding = self.funding_7d_avg[-2]
        
        entry_condition = (
            (prev_close < prev_vwap) and 
            (prev_funding > 2 * prev_avg_funding)
        )
        
        if entry_condition:
            print(f"🚀🌙 ENTRY SIGNAL DETECTED! 🌙🚀")
            
            # Calculate risk parameters 🌑
            entry_price = self.data.Open[-1]
            swing_high = self.swing_high[-1]
            current_vwap = self.weekly_vwap[-1]
            stop_loss = max(swing_high, current_vwap)
            
            risk_per_share = stop_loss - entry_price
            if risk_per_share <= 0:
                print(f"⚠️🌙 Invalid risk calculation - Skipping trade")
                return
                
            risk_amount = self.equity() * self.risk_pct
            position_size = risk_amount / risk_per_share