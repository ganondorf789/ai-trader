```python
# 🌙 Moon Dev's BandedOverbought Backtest Implementation
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# 🚀 Data Preparation Magic
def load_data(data_path):
    # Load and clean data
    data = pd.read_csv(data_path, parse_dates=['datetime'])
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    
    # 🌌 Cosmic Column Alignment
    column_map = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'openinterest': 'OpenInterest'
    }
    data.rename(columns=column_map, inplace=True)
    data.set_index('datetime', inplace=True)
    return data

# 🌗 Strategy Core
class BandedOverbought(Strategy):
    risk_per_trade = 0.01  # 1% of equity per trade
    stop_loss_pct = 0.02   # 2% initial stop loss
    
    def init(self):
        # 🌠 Indicator Constellation
        self.close = self.data.Close
        self.high = self.data.High
        self.oi = self.data.OpenInterest
        
        # 📊 Bollinger Bands Calculation
        self.upper, self.middle, self.lower = self.I(talib.BBANDS, self.close, 
                                                  timeperiod=20, nbdevup=2, nbdevdn=2, 
                                                  name=['UpperBB', 'MiddleBB', 'LowerBB'])
        
        # 🌕 5-Day High Detection
        self.high_5d = self.I(talib.MAX, self.high, timeperiod=480, name='5D_High')
        
        # 💎 Open Interest Analysis
        self.oi_max = self.I(talib.MAX, self.oi, timeperiod=480, name='OI_Max')
        self.oi_pct = self.I(lambda: (self.oi / self.oi_max) * 100, name='OI_Pct')
        
    def next(self):
        price = self.data.Close[-1]
        
        # 🌑 Entry Conditions
        if not self.position:
            # Cosmic alignment conditions
            upper_band_break = self.high[-1] >= self.upper[-1]
            new_5d_high = self.high[-1] >= self.high_5d[-1]
            oi_valid = self.oi_pct[-1] >= 20
            
            if upper_band_break and new_5d_high and oi_valid:
                # 🧮 Risk Management Calculation
                equity = self.equity
                risk_amount = equity * self.risk_per_trade
                stop_price = price * (1 + self.stop_loss_pct)
                risk_per_share = stop_price - price
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.sell(size=position_size, sl=stop_price,
                                tag=f"🌌 SHORT ENTRY | Size: {position_size}")
                        print(f"🌙✨ MOON DEV ALERT: Sold {position_size} @ {price:.2f} | "
                             f"SL: {stop_price:.2f} | Equity: {equity:,.0f} 🚀")

        # 🌕 Exit Conditions
        else:
            # Stellar exit signals
            price_below_upper = self.close[-1] < self.upper[-1]
            oi_collapse = self.oi_pct[-1] < 10
            
            if price_below_upper or oi_collapse:
                self.position.close()
                print(f"🌙💫 MOON DEV EXIT: Closed @ {price:.2f} | "
                     f"Reason: {'Bands' if price_below_upper else 'OI'} | "
                     f"Equity: {self.equity:,.0f} 🌠")

# 🌍 Backtest Launch Sequence
if __name__ == "__main__":
    data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
    data = load_data(data_path)
    
    # 🚀 Initialize Cosmic Backtester
    bt = Backtest(data, BandedOverbought, cash=100000, commission=.002)
    
    # 🌟 Run the Backtest
    stats = bt.run()
    
    # 🌌 Display Results
    print(f"