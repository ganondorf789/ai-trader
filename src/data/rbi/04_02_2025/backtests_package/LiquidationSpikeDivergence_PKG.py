I'll fix the code to remove all `backtesting.lib` dependencies and ensure proper indicator usage. Here's the corrected version:

```python
# 🌙 Moon Dev's Backtest AI Implementation 🚀

import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

# ========== DATA PREPROCESSING ==========
def load_data(path):
    # Load and clean data with Moon Dev standards 🌕
    data = pd.read_csv(path)
    
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    print("🌌 Columns after cleaning:", data.columns.tolist())
    
    # Drop unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col], errors='ignore')
    
    # Map required columns with proper case
    column_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    data.rename(columns=column_mapping, inplace=True)
    
    # Convert and set datetime index
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    
    print("🔭 First 3 cosmic readings:")
    print(data.head(3))
    return data

# ========== STRATEGY IMPLEMENTATION ==========
class LiquidationSpikeDivergence(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌓
    
    def init(self):
        # ===== LIQUIDATION SPIKE DETECTION =====
        self.liquidation = self.I(lambda x: x, self.data.df['liquidation'], name='Liquidation')
        self.liquidation_mean = self.I(talib.SMA, self.liquidation, timeperiod=20, name='Liq Mean')
        self.liquidation_std = self.I(talib.STDDEV, self.liquidation, timeperiod=20, name='Liq STD')
        
        # ===== RSI DIVERGENCE =====
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14, name='RSI 14')
        
        # ===== CHANDE-KROLL STOP =====
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=10, name='ATR 10')
        self.max_high = self.I(talib.MAX, self.data.High, timeperiod=10, name='Max High')
        self.upper_band = self.I(lambda h, a: h - 3*a, self.max_high, self.atr, name='CK Upper')
        
        print("🌠 Moon Dev Indicators Initialized!")
    
    def next(self):
        current_close = self.data.Close[-1]
        current_low = self.data.Low[-1]
        
        # ===== LIQUIDATION SPIKE CHECK =====
        liq_spike = (self.liquidation[-1] > (self.liquidation_mean[-1] + 2*self.liquidation_std[-1]))
        
        # ===== BULLISH DIVERGENCE CHECK =====
        if len(self.data) < 15:  # Ensure enough data for RSI
            return
        
        # Look for lower price low + higher RSI low
        price_lows = self.data.Low[-3:]
        rsi_lows = self.rsi[-3:]
        
        bullish_divergence = (
            (price_lows[-1] < price_lows[-2]) and  # Current lower low
            (rsi_lows[-1] > rsi_lows[-2])          # Current higher RSI low
        )
        
        # ===== ENTRY LOGIC =====
        if not self.position and liq_spike and bullish_divergence:
            # Risk management calculations 🌗
            entry_price = current_close
            atr_value = self.atr[-1]
            
            # Dynamic stop loss below swing low
            stop_loss_price = min(self.data.Low[-5:]) - 0.5*atr_value
            risk_amount = self.risk_per_trade * self.equity
            risk_per_share = entry_price - stop_loss_price
            
            if risk_per_share <= 0:
                print("🚨 Invalid risk calculation - skipping trade")
                return
            
            position_size = int(round(risk_amount / risk_per_share))
            
            if position_size > 0:
                print(f"🚀🌕 MOON SHOT! Long {position_size} @ {