```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR VOLTAICBREAKOUT STRATEGY 🚀

# ===== REQUIRED IMPORTS =====
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import numpy as np

# ===== DATA PREPROCESSING =====
# Load cosmic crypto data 🌌
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Cleanse stardust (data cleaning) ✨
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Align celestial coordinates (column mapping) 🌍
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'datetime': 'DateTime'
}, inplace=True)

# Set temporal index 🕰️
data['DateTime'] = pd.to_datetime(data['DateTime'])
data.set_index('DateTime', inplace=True)

# ===== VOLTAICBREAKOUT STRATEGY CLASS =====
class VoltaicBreakout(Strategy):
    # 🌙 MOON DEV CONFIGURATIONS
    risk_per_trade = 0.01  # 1% risk per trade
    adx_threshold = 30
    rsi_exit_level = 70
    
    def init(self):
        # ===== CELESTIAL INDICATORS =====
        # Cosmic volatility bands (Bollinger Bands) 🌗
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, name='BB_UPPER', which=0)
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, name='BB_LOWER', which=2)
        
        # Stellar trend strength (ADX) 🌠
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ADX_14')
        
        # Momentum pulsar (RSI) ⏳
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14, name='RSI_14')

    def next(self):
        # ===== GALACTIC POSITION MANAGEMENT =====
        price = self.data.Close[-1]
        
        # 🌙 LONG ENTRY: Breakout with cosmic confirmation
        if not self.position:
            if (self.data.Close[-2] <= self.bb_upper[-2] and
                price > self.bb_upper[-1] and
                self.adx[-1] > self.adx_threshold):
                
                # Calculate moon-powered position size 🌚
                stop_loss = self.bb_lower[-1]
                risk_amount = self.risk_per_trade * self.equity
                risk_per_share = price - stop_loss
                
                if risk_per_share > 0:  # Avoid black hole trades
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.buy(size=position_size, sl=stop_loss)
                        print(f"🌙✨ COSMIC BREAKOUT! Long {position_size} units at {price:.2f} | ADX: {self.adx[-1]:.1f} 🚀")
        
        # 🪐 EXIT CONDITION: Momentum decay detected
        elif self.position.is_long:
            if self.rsi[-1] < self.rsi_exit_level and self.rsi[-2] >= self.rsi_exit_level:
                self.position.close()
                print(f"🌑 RSI COLLAPSE! Exiting at {price:.2f} | Equity: {self.equity:.2f} ⚠️")

# ===== LAUNCH BACKTEST ENGINE =====
print("\n🚀🌙 INITIATING MOON DEV BACKTEST PROTOCOL 🌙🚀")
bt = Backtest(data, VoltaicBreakout, cash=1_000_000, exclusive_orders=True)

# ===== EXECUTE COSMIC SIMULATION =====
stats = bt.run()
print("\n🌌🌌🌌 FINAL MISSION STATISTICS 🌌🌌🌌")
print(stats)
print(stats