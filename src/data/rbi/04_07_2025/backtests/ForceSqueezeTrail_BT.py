```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION FOR ForceSqueezeTrail STRATEGY 🚀

import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# =====================
# DATA PREPARATION 🌙✨
# =====================
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'])

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data.set_index('datetime', inplace=True)

# =====================
# STRATEGY CLASS 🌙📈
# =====================
class ForceSqueezeTrail(Strategy):
    def init(self):
        # 🌙✨ INDICATORS USING TA-LIB AND self.I() WRAPPER
        # Elder Force Index (13-period)
        self.efi = self.I(talib.FORCEINDEX, self.data.Close, self.data.Volume, timeperiod=13, name='EFI')
        
        # Bollinger Bands (20,2) with Bandwidth
        def bb_bandwidth(close):
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return (upper - lower) / middle
        self.bb_width = self.I(bb_bandwidth, self.data.Close, name='BB_Width')
        self.bb_width_sma = self.I(talib.SMA, self.bb_width, timeperiod=20, name='BB_Width_SMA')
        
        # ATR (14-period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR')
        
        # 🌙 TRADE TRACKING VARIABLES
        self.entry_bar = None
        self.highest_high = None

    def next(self):
        current_bar = len(self.data) - 1
        
        # 🌙✨ MOON DEV DEBUG PRINTS
        if current_bar % 100 == 0:
            print(f"🌙 Bar {current_bar} | Close: {self.data.Close[-1]} | EFI: {self.efi[-1]:.2f} | BB Width: {self.bb_width[-1]:.4f} | ATR: {self.atr[-1]:.2f}")

        # Skip if not enough data
        if len(self.efi) < 2 or len(self.bb_width_sma) < 1:
            return

        # =====================
        # ENTRY LOGIC 🌙🚀
        # =====================
        if not self.position:
            # EFI cross above zero
            efi_cross = self.efi[-1] > 0 and self.efi[-2] <= 0
            
            # Bollinger contraction condition
            bb_contraction = (self.bb_width[-1] < self.bb_width_sma[-1]) or (self.bb_width[-1] < 0.5)
            
            if efi_cross and bb_contraction:
                # 🌙 RISK MANAGEMENT CALCULATIONS
                risk_percent = 0.01  # 1% risk per trade
                risk_amount = self.equity * risk_percent
                entry_price = self.data.Close[-1]
                atr_value = self.atr[-1]
                
                # Position sizing
                stop_loss = entry_price - atr_value  # 1x ATR initial stop
                risk_per_share = entry_price - stop_loss
                if risk_per_share <= 0:
                    return  # Skip invalid calculation
                
                position_size = int(round(risk_amount / risk_per_share))
                if position_size > 0:
                    self.buy(size=position_size)
                    self.entry_bar = current_bar
                    self.highest_high = self.data.High[-1]
                    print(f"\n🚀🌙 ENTRY | Price: {entry_price} | Size: {position_size} | SL: {stop_loss:.1f}")

        # =====================
        # EXIT LOGIC 🌙🔒