```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION FOR VolatilityYieldDivergence STRATEGY 🚀

# ================
# IMPORTS
# ================
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np
import pandas_ta as ta

# ================
# DATA PREPARATION
# ================
def prepare_data(path):
    # Load data with Moon Dev precision 🌙
    data = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
    
    # Cleanse columns with lunar magic ✨
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    
    # Align celestial bodies with proper case mapping 🌍
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    return data

# ======================
# STRATEGY IMPLEMENTATION
# ======================
class VolatilityYieldDivergence(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌕
    stop_loss_pct = 0.01  # 1% stop loss 🛑
    
    def init(self):
        # 🌗 CELESTIAL INDICATORS CALCULATION
        # Volatility proxy (ATR)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR_14')
        
        # Price trend (20-period SMA)
        self.price_sma = self.I(talib.SMA, self.data.Close, 20, name='SMA_20')
        
        # Yield proxy (20-period EMA)
        self.yield_ema = self.I(talib.EMA, self.data.Close, 20, name='EMA_20')
        
        # Correlation between volatility and yields
        self.corr = self.I(talib.CORREL, self.atr, self.yield_ema, 20, name='CORREL_20')
        
        # Momentum indicators 🌊
        self.rsi = self.I(talib.RSI, self.data.Close, 14, name='RSI_14')
        self.macd, self.macdsignal, _ = self.I(talib.MACD, self.data.Close, 12, 26, 9, name='MACD')

    def next(self):
        # Wait for full moon phase (sufficient data) 🌕
        if len(self.data) < 20:
            return

        # 🌙✨ ENTRY CONDITIONS ✨🌙
        volatility_rising = self.atr[-1] > self.atr[-2]
        price_stagnant = self.price_sma[-1] < self.price_sma[-2]
        inverse_correlation = self.corr[-1] < 0 and self.corr[-1] < self.corr[-2]
        
        # Replace crossover with manual check
        momentum_buy = (self.rsi[-1] < 30) or (self.macd[-2] < self.macdsignal[-2] and self.macd[-1] > self.macdsignal[-1])

        if volatility_rising and price_stagnant and inverse_correlation and momentum_buy:
            # 🌑 MOON-SIZED POSITION CALCULATION
            price = self.data.Close[-1]
            stop_price = price * (1 - self.stop_loss_pct)
            risk_amount = self.equity * self.risk_percent
            position_size = int(round(risk_amount / (price - stop_price)))
            
            if position_size > 0:
                self.buy(size=position_size, sl=stop_price)
                print(f"🌙🚀 MOON DEV ALERT: LONG {position_size} @ {price:.2f} | SL {stop_price:.2f} 🚀🌙")

        # 🌙✨ EXIT CONDITIONS ✨🌙
        for trade in self.trades:
            if trade.is_long:
                # Exit on RSI overbought or trend reversal
                if self.rsi[-1] > 70 or self.data.Close[-1] < self.price_sma[-1]:
                    self.sell()
                    print(f"🌙🌌 COSMIC EXIT: {self.data.Close[-1]:.2f} | PROFIT {trade.pl:.2f} 🌌🌙")

# ================
# BACKTEST EXECUTION
# ================
if __name