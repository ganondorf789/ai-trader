```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION FOR VOLATILITY COMPRESSION FORCE STRATEGY 🚀

# Import necessary libraries
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import numpy as np

# Data preprocessing moon-style 🌕✨
def load_data(path):
    # Load and clean data
    data = pd.read_csv(
        path,
        parse_dates=['datetime'],
        index_col='datetime'
    )
    
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    
    # Proper column mapping
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    return data

# Load moon data 🌙📈
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = load_data(data_path)

class VolatilityCompressionForce(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌕🛡️
    
    def init(self):
        # 🌙 ELDER FORCE INDEX CALCULATION
        self.efi = self.I(ta.fi, 
                         self.data.Close, 
                         self.data.Volume, 
                         length=13, 
                         name='MOON_EFI_13')
        
        # 🎯 BOLLINGER BANDS CALCULATION
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS,
            self.data.Close,
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
            matype=0,
            name=['MOON_BB_UP', 'MOON_BB_MID', 'MOON_BB_LOW']
        )
        
        # 📏 BOLLINGER BAND WIDTH
        self.bb_width = self.I(
            lambda: (self.bb_upper - self.bb_lower) / self.bb_middle,
            name='MOON_BB_WIDTH'
        )
        
        # 📈 HISTORICAL PERCENTILE CALCULATION (1-year rolling)
        self.bb_width_percentile = self.I(
            lambda: self.bb_width.rolling(35040).quantile(0.2).ffill(),
            name='MOON_BB_PCTL'
        )
        
        # 🌪️ ATR FOR POSITION SIZING
        self.atr = self.I(talib.ATR,
                         self.data.High,
                         self.data.Low,
                         self.data.Close,
                         timeperiod=14,
                         name='MOON_ATR_14')
        
        # ⏳ TRADE ENTRY TIME TRACKING
        self.entry_time = None

    def next(self):
        current_close = self.data.Close[-1]
        moon_equity = self.equity  # 🌕💵
        
        # 🌙 EXIT CONDITIONS
        if self.position:
            # Time-based exit (48 hours = 192 15m periods)
            if len(self.data) - self.position.entry_bar >= 192:
                self.position.close()
                print(f"🌙⏰ MOON TIMEOUT EXIT | Equity: {moon_equity:,.2f}")
                
        # 🚀 ENTRY CONDITIONS
        if not self.position:
            # Volatility compression check
            volatility_cond = self.bb_width[-1] < self.bb_width_percentile[-1]
            
            # Momentum confirmation
            momentum_cond = crossover(self.efi, 0)
            
            if volatility_cond and momentum_cond:
                # 🌕 RISK MANAGEMENT CALCULATIONS
                atr_value = self.atr[-1]
                if atr_value <= 0:
                    return  # Avoid division by zero
                
                # Calculate position size with moon math 🌙🧮
                risk_amount = self.risk_percent * moon_equity
                position_size = int(round(risk_amount / atr_value))
                
                if position_size > 0:
                    # 🚀 ENTRY WITH DYNAMIC STOPS
                    self.buy(
                        size=position_size,
                        sl=self.bb_lower[-1],
                        tp=self.bb_upper[-1]
                    )
                    print(f"🌕🚀 MOON ENTRY | Size