I'll fix the code by removing all `backtesting.lib` dependencies and ensuring proper indicator usage. Here's the corrected version with Moon Dev themed debug prints:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 MOON DEV DATA PREPARATION 🌙
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and prepare data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class LiquidationPulse(Strategy):
    risk_pct = 0.02
    atr_multiplier = 1.5
    oi_lookback = 16  # 4 hours in 15m intervals
    
    def init(self):
        # 🌙 CORE INDICATORS ✨
        # Funding Rate Analysis
        self.funding_rate = self.data.df['funding_rate']
        self.funding_high = self.I(lambda x: x.rolling(2880).quantile(0.8),
                                  self.funding_rate, name='FUNDING_HIGH')
        
        # Open Interest Momentum
        self.open_interest = self.data.df['open_interest']
        self.oi_pct_change = self.I(lambda x: x.pct_change(self.oi_lookback),
                                   self.open_interest, name='OI_CHANGE')
        
        # Ichimoku Cloud
        self.tenkan, self.kijun, self.senkou_a, self.senkou_b = self.I(
            talib.ICHIMOKU, self.data.High, self.data.Low, 
            9, 26, 52, name=['TENKAN', 'KIJUN', 'SENKOU_A', 'SENKOU_B']
        )
        
        # Volatility System
        self.atr_1h = self.I(talib.ATR, self.data.High, self.data.Low, 
                            self.data.Close, 56, name='ATR_1H')
        self.atr_ma = self.I(talib.SMA, self.atr_1h, 32, name='ATR_MA')
        
        # Risk Management
        self.atr_4h = self.I(talib.ATR, self.data.High, self.data.Low,
                            self.data.Close, 16, name='ATR_4H')
        self.prev_4h_high = self.I(talib.MAX, self.data.High, 16, name='4H_HIGH')

    def next(self):
        # 🌙 MOON DEV TRADE LOGIC ✨
        if len(self.data) < 2880:  # Ensure full data window
            return
            
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        
        # Entry Conditions
        if not self.position:
            funding_cond = self.funding_rate[-1] >= self.funding_high[-1]
            oi_cond = self.oi_pct_change[-1] <= -0.15
            senkou_a_val = self.senkou_a[-26]  # Adjusted for forward shift
            senkou_b_val = self.senkou_b[-26]
            price_cond = (current_close < senkou_a_val) & (current_close < senkou_b_val)
            
            if funding_cond and oi_cond and price_cond:
                # Risk Calculation
                equity = self.equity
                risk_amount = equity * self.risk_pct
                atr_val = self.atr_4h[-1]
                stop_price = current_close + (atr_val * self.atr_multiplier)
                risk_per_share = stop_price - current_close
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    self.sell(size=position_size, sl=stop_price)
                    self.entry_4h_high = self.prev_4h_high[-1]
                    
                    print(f"🌙✨ MOON DEV SHORT ACTIVATED ✨🌙")
                    print(f"📊 Entry: {current_close:.2f} | 📦 Size: {position_size}")
                    print