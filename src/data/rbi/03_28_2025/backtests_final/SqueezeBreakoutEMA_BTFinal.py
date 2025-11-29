I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete corrected version with Moon Dev themed debug prints:

```python
# backtesting_squeeze_breakout_ema.py
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

def load_data(path):
    """🌙 Load and prepare Moon Dev's sacred market data ✨"""
    data = pd.read_csv(path)
    
    # Cleanse the data temple 🕍
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    
    # Align cosmic columns 🌌
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    
    # Set temporal coordinates ⏳
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

class SqueezeBreakoutEMA(Strategy):
    def init(self):
        """🌠 Initialize celestial indicators"""
        # EMA Convergence Matrix 🌗
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=50, name='EMA50')
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=200, name='EMA200')
        
        # Bollinger Singularity Field 📡
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, 
            matype=0, name=['BB_Upper', 'BB_Mid', 'BB_Lower']
        )
        
        # Quantum Squeeze Detector 🔭
        bb_width = self.bb_upper - self.bb_lower
        self.bb_width = self.I(bb_width, name='BB_Width')
        self.min_width = self.I(talib.MIN, self.bb_width, timeperiod=50, name='Min_Width')
        
        # Volume Vortex Indicator 🌪️
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='Vol_MA')
        
        # Gravitational Swing Levels ⚖️
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='Swing_Low')
        
        print("🌙 Moon Dev systems online! Ready to ride the breakout waves! 🏄♂️")

    def next(self):
        """🚀 Execute cosmic trading protocol"""
        # Wait for star alignment 🌟
        if len(self.data.Close) < 200:
            return

        # Current constellation data 🌠
        price = self.data.Close[-1]
        ema50 = self.ema50[-1]
        ema200 = self.ema200[-1]
        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]
        vol = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1] or 1  # Prevent black hole division

        # Cosmic Alignment Check 🌑🌕
        golden_cross = (self.ema50[-2] < self.ema200[-2] and self.ema50[-1] > self.ema200[-1])
        squeeze = (self.bb_width[-1] <= self.min_width[-1] * 1.2)
        breakout = price > bb_upper
        vol_spike = vol >= 2 * vol_ma

        # ENTRY: Stargate Activation Sequence 🌌
        if not self.position and all([golden_cross, squeeze, breakout, vol_spike]):
            # Calculate meteor deflection parameters ☄️
            risk_pct = 0.01  # 1% of cosmic reserves
            sl_price = min(self.swing_low[-1], bb_lower)
            risk_per_share = price - sl_price
            
            if risk_per_share <= 0:
                print("🌙⚠️ Cosmic anomaly detected! Aborting launch.")
                return
                
            position_size = int(round((self.equity * risk_pct) / risk_per_share))
            tp_price = price + 2 * (price - sl_price)  # Fixed TP calculation
            
            # Engage hyperdrive 🚀
            self.buy(size=position_size, sl=sl_price, tp=tp_price)
            print(f"🌙✨ Starg