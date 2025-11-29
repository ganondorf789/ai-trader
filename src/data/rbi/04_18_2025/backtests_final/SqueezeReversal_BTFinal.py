Here's the complete fixed backtest code with Moon Dev themed debug prints and proper position sizing implementation:

```python
# 🌙 MOON DEV BACKTESTING IMPLEMENTATION 🌙
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

def load_data(data_path):
    """🌌 Load and prepare cosmic data with Moon Dev standards 🌌"""
    data = pd.read_csv(data_path)
    
    # 🧹 Cleanse column names
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    
    # 🌠 Align with celestial column mapping
    column_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    data.rename(columns=column_mapping, inplace=True, errors='ignore')
    
    # 🕰 Convert time dimension
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

class SqueezeReversal(Strategy):
    max_positions = 3  # 🚀 Maximum concurrent launches
    risk_pct = 0.01    # 🌑 1% risk per trade
    
    def init(self):
        """🌓 Calculate celestial indicators"""
        # 🌗 Core strategy indicators
        self.sma20 = self.I(talib.SMA, self.data.Close, timeperiod=20, name='SMA20')
        self.rsi14 = self.I(talib.RSI, self.data.Close, timeperiod=14, name='RSI14')
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR14')
        
        # 🌌 Funding and OI indicators
        self.oi_low = self.I(talib.MIN, self.data.df['openinterest'], timeperiod=96, name='OI_LOW_24H')
        self.funding_series = self.data.df['fundingrate']
        
    def next(self):
        """🌕 Execute cosmic trading logic"""
        if len(self.data) < 100:  # Wait for stardust to settle
            return
            
        # 🌠 Current constellation values
        current_close = self.data.Close[-1]
        current_funding = self.funding_series[-1]
        prev_funding = self.funding_series[-2] if len(self.funding_series) > 1 else 0
        current_oi = self.data.df['openinterest'][-1]
        oi_low = self.oi_low[-1]
        
        # 🌑 Entry conditions
        funding_trigger = (current_funding < -0.001) and (prev_funding >= -0.001)
        oi_trigger = (current_oi > oi_low * 1.15) if oi_low > 0 else False
        trend_confirm = current_close < self.sma20[-1]
        momentum_confirm = self.rsi14[-1] < 40
        
        # 🚀 Launch sequence check
        if (funding_trigger and oi_trigger and trend_confirm and momentum_confirm 
            and len(self.trades) < self.max_positions and not self.position):
            
            # 🌑 Calculate cosmic risk parameters
            atr = self.atr14[-1]
            if atr == 0:  # Protect against black holes
                return
                
            risk_amount = self.risk_pct * self.equity
            position_size = int(round(risk_amount / (2 * atr)))
            
            if position_size > 0:
                # 🌠 Execute launch sequence
                entry_price = self.data.Close[-1]
                stop_loss = entry_price - 2 * atr
                take_profit = entry_price + 1.5 * atr
                
                self.buy(
                    size=position_size,
                    sl=stop_loss,
                    tp=take_profit,
                    tag={
                        'entry_price': entry_price,
                        'initial_atr': atr,
                        'trailing_active': False
                    }
                )
                print(f"🚀🌑 MOON DEV LAUNCH: Long {position_size} units @ {entry_price:.2f} | SL: {stop_loss:.2f} TP: {take_profit:.2f}")
        
        # 🌓 Manage active missions
        for trade in self.trades:
            if not trade.is_long:
                continue
                
            tag = trade.tag
            current_high = self.data