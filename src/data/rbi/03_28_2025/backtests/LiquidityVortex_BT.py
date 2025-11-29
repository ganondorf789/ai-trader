```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR LIQUIDITYVORTEX STRATEGY 🚀

import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

# =====================
# DATA PREPARATION 🌌
# =====================
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

# Map to backtesting.py format
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'funding_rate': 'FundingRate'  # Assumes funding rate column exists
})

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

# =====================
# STRATEGY CORE 🌪️✨
# =====================
class LiquidityVortex(Strategy):
    # Strategy parameters
    atr_period = 14
    swing_window = 20
    funding_threshold = 0.0005  # 0.05% for BTC
    risk_pct = 0.01  # 1% risk per trade
    volatility_multiplier = 1.5
    
    def init(self):
        # 🌙 Indicators using TA-Lib through self.I()
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_window)
        
        # Funding rate data (raw column)
        self.funding_series = self.data['FundingRate']
        
        print("🌙✨ LIQUIDITY VORTEX INITIALIZED! Ready to hunt market inefficiencies 🌪️")

    def next(self):
        # Skip weekends (Moon Dev's Time Filter 🌗)
        if self.data.index[-1].weekday() >= 5:
            return

        # Get current values
        price = self.data.Close[-1]
        funding = self.funding_series[-1]
        current_swing_high = self.swing_high[-1]
        current_swing_low = self.swing_low[-1]
        current_atr = self.atr[-1]

        # =====================
        # ENTRY CONDITIONS 🚀
        # =====================
        if not self.position:
            # Funding rate divergence 🌊
            funding_condition = abs(funding) >= self.funding_threshold
            
            # Liquidity void detection (2% gap from swing levels) 🕳️
            liquidity_condition = (
                (self.data.High[-1] > current_swing_high * 1.02) or 
                (self.data.Low[-1] < current_swing_low * 0.98
            ))
            
            # Volatility spike (1.5x ATR) ⚡
            volatility_condition = (self.data.High[-1] - self.data.Low[-1]) > self.volatility_multiplier * current_atr

            if funding_condition and liquidity_condition and volatility_condition:
                # Moon Dev's Risk Calculation 🌙
                risk_amount = self.equity * self.risk_pct
                
                if funding > 0:  # Short entry (overcrowded longs)
                    stop_loss = self.data.High[-1] + (current_atr * 1.5)
                    risk_distance = stop_loss - price
                    position_size = int(round(risk_amount / risk_distance))
                    
                    if position_size > 0:
                        self.sell(size=position_size, sl=stop_loss, tp=price - (2 * risk_distance))
                        print(f"🌙🚀 SHORT ENTRY! Price: {price:.2f} | Size: {position_size}")
                        print(f"    SL: {stop_loss:.2f} | Risk/Reward: 1:2 🌪️")
                else:  # Long entry (overcrowded shorts)
                    stop_loss = self.data.Low[-1] - (current_atr * 1.5)
                    risk_distance = price - stop_loss
                    position_size = int(round(risk_amount / risk_distance))
                    
                    if position_size > 0:
                        self