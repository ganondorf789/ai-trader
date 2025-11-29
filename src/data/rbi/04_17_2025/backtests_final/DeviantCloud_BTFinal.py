Here's the debugged backtest code with Moon Dev themed improvements while maintaining the original strategy logic:

```python
# -*- coding: utf-8 -*-
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import pandas_ta as ta

class DeviantCloud(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌙 MOON DEV DATA PREPARATION PHASE 🌙
        # Ichimoku Cloud Calculation
        high = self.data.High
        low = self.data.Low
        
        # Calculate Ichimoku components using pandas_ta
        ichimoku = ta.ichimoku(high, low, tenkan=9, kijun=26, senkou=52)
        self.tenkan = self.I(ichimoku['ITS_9'], name='Tenkan')
        self.kijun = self.I(ichimoku['IKS_26'], name='Kijun')
        self.senkou_a = self.I(ichimoku['ISA_9'], name='Senkou A')
        self.senkou_b = self.I(ichimoku['ISB_26'], name='Senkou B')
        
        # Funding Rate Analysis
        if 'funding_rate' not in self.data.df.columns:
            raise ValueError("🌑 CRITICAL: Funding rate data missing from DataFrame")
            
        funding_series = self.data.df['funding_rate']
        self.funding_mean = self.I(lambda x: funding_series.rolling(2880).mean(), name='Fund30MA')  # 30-day (2880*15m)
        self.funding_std = self.I(lambda x: funding_series.rolling(2880).std(), name='Fund30Std')
        self.funding_upper = self.I(lambda x: self.funding_mean + 2*self.funding_std, name='FundUpper')
        
    def next(self):
        price = self.data.Close[-1]
        
        # 🌙✨ ENTRY LOGIC PHASE ✨🌙
        if not self.position:
            # Funding rate extreme condition
            fund_condition = (self.data.df['funding_rate'].iloc[-1] > self.funding_upper[-1])
            
            # Ichimoku cloud condition
            cloud_top = max(self.senkou_a[-1], self.senkou_b[-1])
            cloud_condition = (price < cloud_top)
            
            if fund_condition and cloud_condition:
                # 🚀 RISK MANAGEMENT CALCULATIONS 🚀
                stop_loss = cloud_top
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = abs(stop_loss - price)
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:  # Ensure valid position size
                        self.sell(size=position_size, sl=stop_loss, tag="moon_short")
                        print(f"🌙✨ MOON DEV SHORT ACTIVATED ✨🌙")
                        print(f"| Entry: {price:.2f} | SL: {stop_loss:.2f}")
                        print(f"| Size: {position_size} units | Risk: {self.risk_per_trade*100:.1f}% Equity")
        
        # 🌑 EXIT LOGIC PHASE 🌑
        elif self.position:
            # Mean reversion exit condition
            if self.data.df['funding_rate'].iloc[-1] <= self.funding_mean[-1]:
                self.position.close()
                print(f"🚀🌑 MOON DEV EXIT SIGNAL 🌑🚀")
                print(f"| Exit: {price:.2f} | PnL: {self.position.pl:.2f}")

# 🌙 DATA PREPROCESSING FOR MOON DEV SYSTEMS 🌙
try:
    data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
    
    # Clean columns
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    
    # Ensure proper column mapping
    required_columns = {'open', 'high', 'low', 'close', 'volume', 'funding_rate', 'datetime'}
    if not required_columns.issubset(data.columns):
        missing = required_columns - set(data.columns)
        raise ValueError(f"🌑 CRITICAL: Missing required columns: {missing}")
        
    data = data.rename