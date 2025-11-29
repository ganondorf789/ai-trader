```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder
import numpy as np

# Load and preprocess data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

# Rename columns to backtesting.py format
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class LiquidationReversion(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    funding_rate_threshold = -0.001  # -0.1% 
    funding_rate_normal = -0.0005    # Normalization threshold
    bb_period = 20
    bb_std = 2
    swing_period = 20  # Swing low lookback
    
    def init(self):
        # Precompute indicators
        close = self.data.Close
        
        # Bollinger Bands using TA-Lib 🌗
        upper, self.middle, lower = talib.BBANDS(close, 
            timeperiod=self.bb_period, 
            nbdevup=self.bb_std, 
            nbdevdn=self.bb_std, 
            matype=0
        )
        self.upper = self.I(lambda: upper, name='UpperBB')
        self.lower = self.I(lambda: lower, name='LowerBB')
        
        # Swing low for stop loss 📉
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period)
        
        # Access funding rate data 🌊
        self.funding_rates = self.data.df['funding_rate'].values
        
        print("🌙 Lunar Strategy Activated! Tracking BTC Short Squeezes 🚀")
        print(f"🔮 Config | Risk: {self.risk_per_trade*100}% | BB: {self.bb_period}/{self.bb_std} | FR Threshold: {self.funding_rate_threshold*100}%")
    
    def next(self):
        current_idx = len(self.data) - 1
        current_funding = self.funding_rates[current_idx] if current_idx < len(self.funding_rates) else 0
        
        if self.position:
            # Exit conditions 🌈
            price = self.data.Close[-1]
            
            # Take profit at middle BB 🎯
            if price >= self.middle[-1]:
                self.position.close()
                print(f"🚀✨ TP Reached! Exited at {price:.2f} (Middle BB)")
            
            # Funding rate normalization 🌀
            elif current_funding >= self.funding_rate_normal:
                self.position.close()
                print(f"🌊🔄 Funding Normalized! Exited at {price:.2f}")
            
            # Stop loss check 🛑
            elif self.data.Low[-1] < self.swing_low[-1]:
                self.position.close()
                print(f"🌪️💥 SL Triggered! Exited at {self.data.Low[-1]:.2f}")
        else:
            # Entry conditions 🚪
            entry_condition = (
                current_funding < self.funding_rate_threshold and 
                self.data.Close[-1] < self.lower[-1]
            )
            
            if entry_condition:
                sl_price = self.swing_low[-1]
                entry_price = self.data.Close[-1]
                risk_amount = self.equity * self.risk_per_trade
                price_diff = entry_price - sl_price
                
                if price_diff > 0:
                    position_size = risk_amount / price_diff
                    position_size = int(round(position_size))
                    
                    if position_size > 0:
                        self.buy(size=position_size, sl=sl_price)
                        print(f"🌙🚀 LONG ENTRY! Size: {position_size} | Entry: {entry_price:.2f} | SL: {sl_price:.2f}")
                    else:
                        print("⚠️🌙 Insufficient Size - Trade Skipped")
                else:
                    print(f"⚠️🌙 Invalid SL | Entry: {entry_price:.2f} | SL: {sl_price:.2f}")

# Execute backtest 🌕
bt = Backtest(data, LiquidationReversion, cash=1_000_000, commission=.002)
stats = bt