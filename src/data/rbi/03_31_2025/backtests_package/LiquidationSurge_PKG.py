Here's the fixed code with all backtesting.lib imports removed and proper indicator implementations:  

```python
# 🌙 MOON DEV BACKTESTING SCRIPT - LIQUIDATIONSURGE STRATEGY 🚀

from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# =====================
# DATA PREPROCESSING 🌌
# =====================
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping ✨
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# =====================
# STRATEGY CLASS 🚀🌙
# =====================
class LiquidationSurge(Strategy):
    # Risk parameters 🌊
    risk_per_trade = 0.01  # 1% risk per trade
    atr_period = 14
    cluster_period = 20
    volume_ma_period = 20
    
    def init(self):
        # 🌙 LIQUIDATION HEATMAP (Approximated with MAX/MIN clusters)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.cluster_period, name='SWING HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.cluster_period, name='SWING LOW')
        
        # 📈 VOLUME DIVERGENCE SURGE
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_ma_period, name='VOL MA')
        
        # 🚨 OBV REVERSION
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume, name='OBV')
        self.obv_ema = self.I(talib.EMA, self.obv, timeperiod=9, name='OBV EMA')
        
        # 🌪️ VOLATILITY CONTRACTION (ATR)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period, name='ATR')
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=5, name='ATR MA')
        
        # 🎯 TREND ALIGNMENT (EMA)
        self.ema_fast = self.I(talib.EMA, self.data.Close, timeperiod=5, name='EMA 5')
        
        print("🌙 MOON DEV STRATEGY INITIALIZED! READY FOR LIFTOFF! 🚀")

    def next(self):
        current_price = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        
        # Avoid low liquidity periods (weekends) 🌙
        if pd.Timestamp(self.data.index[-1]).dayofweek >= 5:  # 5=Saturday, 6=Sunday
            return
        
        # =====================
        # ENTRY LOGIC 🌟
        # =====================
        if not self.position:
            # Liquidation cluster proximity (1% from swing high) 🌌
            cluster_upper = self.swing_high[-1]
            cluster_proximity = 0.01 * cluster_upper
            
            # Volume surge condition (2x MA) 🚀
            volume_surge = current_volume > 2 * self.volume_ma[-1]
            
            # EMA trend alignment 📈
            trend_aligned = current_price > self.ema_fast[-1]
            
            if (current_price >= cluster_upper - cluster_proximity and
                volume_surge and
                trend_aligned):
                
                # 🌙 RISK CALCULATION
                sl_distance = 1.5 * self.atr[-1]
                risk_amount = self.equity * self.risk_per_trade
                position_size = int(round(risk_amount / sl_distance))
                
                # 🚀 ENTRY WITH MOON DEV FLAIR!
                self.buy(size=position_size, 
                         sl=current_price - sl_distance