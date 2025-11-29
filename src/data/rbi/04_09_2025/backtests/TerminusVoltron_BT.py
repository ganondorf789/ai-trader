```python
# 🌙 Moon Dev Backtest AI Generated Code - TerminusVoltron Strategy 🌙
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import TrailingStopLoss, crossover

# ========================
# DATA PREPARATION
# ========================
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and prepare data columns 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# ========================
# TERMINUSVOLTRON STRATEGY
# ========================
class TerminusVoltron(Strategy):
    adx_period = 14
    risk_pct = 0.05  # 5% of equity per trade
    trailing_stop = 0.15  # 15% trailing stop
    adx_exit_threshold = 5  # ADX drop of 5 points from peak
    emergency_exit_pct = 0.3  # 30% drop in VIX
    
    def init(self):
        # 🌙 CORE INDICATORS
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, self.adx_period)
        self.di_plus = self.I(talib.PLUS_DI, self.data.High, self.data.Low, self.data.Close, self.adx_period)
        self.di_minus = self.I(talib.MINUS_DI, self.data.High, self.data.Low, self.data.Close, self.adx_period)
        
        # 🌙 VIX TERM STRUCTURE DATA
        # (Assuming data contains 'vix_front' and 'vix_second' columns)
        self.peak_adx = 0
        self.entry_vix = 0
        
    def next(self):
        if len(self.adx) < 2:
            return  # 🌙 Wait for indicators to warm up
        
        # 🌙 MOON DEV SIGNAL CALCULATIONS
        current_close = self.data.Close[-1]
        current_adx = self.adx[-1]
        previous_adx = self.adx[-2]
        adx_rising = current_adx > previous_adx
        
        # 🌙 VIX TERM STRUCTURE ANALYSIS
        front_month = self.data['vix_front'][-1]
        second_month = self.data['vix_second'][-1]
        backwardation = front_month > second_month
        contango = front_month < second_month
        
        # 🌙 DEBUG PRINTING
        print(f"\n🌙 MOON DEV DATA: Close={current_close:.2f} | ADX={current_adx:.2f}")
        print(f"✨ VIX Term Structure: Front={front_month:.2f} vs Second={second_month:.2f}")

        if not self.position:
            # 🌙 ENTRY LOGIC
            if backwardation and current_adx > 25 and adx_rising:
                size = (self.equity * self.risk_pct) / current_close
                size = int(round(size))
                if size > 0:
                    self.buy(size=size, sl=TrailingStopLoss(self.trailing_stop))
                    self.peak_adx = current_adx
                    self.entry_vix = front_month
                    print(f"🚀 MOON DEV ENTRY: Long {size} units @ {current_close:.2f}!")
        else:
            # 🌙 UPDATE PEAK ADX
            if current_adx > self.peak_adx:
                self.peak_adx = current_adx
                print(f"📈 New ADX Peak: {self.peak_adx:.2f}")

            # 🌙 EXIT CONDITIONS
            exit_triggers = []
            
            # 1. Contango Flip
            if contango:
                exit_triggers.append("Contango")
                print("🛑 MOON DEV EXIT: Term structure flipped to contango!")
                
            # 2. ADX Decline
            if current_adx <= (self.peak_adx - self.adx_exit_threshold):
                exit_triggers.append(f"ADX