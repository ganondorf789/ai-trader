```python
# backtesting_funding_band_breach.py
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Moon Dev data preprocessing 🌙
def load_data(path):
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
        'volume': 'Volume',
        'funding_rate': 'Funding_Rate'  # Must exist in data
    })
    
    return data

class FundingBandBreach(Strategy):
    risk_pct = 0.05  # Max 5% risk per trade 🌕
    
    def init(self):
        # Bollinger Bands (20,2) 🌗
        self.upper, self.mid, self.lower = self.I(talib.BBANDS, 
            self.data.Close, 
            timeperiod=20, 
            nbdevup=2, 
            nbdevdn=2, 
            matype=0
        )
        
        # Funding Rate indicators 🌊
        self.funding_sma = self.I(talib.SMA,
            self.data.Funding_Rate,
            timeperiod=2880  # 30 days in 15m intervals
        )
        
        self.lookback = 2880  # 30 days lookback
        print("✨ Strategy initialized with Moon Dev precision! ✨")

    def next(self):
        price = self.data.Close[-1]
        funding = self.data.Funding_Rate[-1]
        
        # Only trade when we have sufficient historical data 🌙
        if len(self.data) < self.lookback:
            return
            
        # Calculate funding rate percentile (5th) 🌌
        hist_funding = self.data.Funding_Rate[-self.lookback:]
        pct_5 = np.percentile(hist_funding, 5)
        
        # Entry conditions 🌗
        if not self.position:
            # Funding rate in extreme & price below lower band 🚨
            if (funding <= pct_5) and (price < self.lower[-1]):
                # Calculate position size based on band width 🌊
                upper = self.upper[-1]
                lower = self.lower[-1]
                bandwidth = upper - lower
                
                if bandwidth <= 0:
                    print("⚠️ Zero bandwidth detected! Cosmic anomaly avoided.")
                    return
                
                # Risk-based position sizing 🌕
                risk_amount = self.risk_pct * self.equity
                position_size = risk_amount / (2 * bandwidth)
                position_size = int(round(position_size))
                
                if position_size > 0:
                    # Enter short with stop loss 🌙
                    stop_price = price + 2 * bandwidth
                    self.sell(
                        size=position_size,
                        sl=stop_price,
                        tag="MoonDevShort"
                    )
                    print(f"🚀🌙 SHORT SIGNAL! Size: {position_size}")
                    print(f"   Entry: {price:.2f}, Stop: {stop_price:.2f}")
                    print(f"   Funding: {funding:.6f} ≤ 5% ({pct_5:.6f})")
        else:
            # Exit conditions 🌗
            current_position = self.position
            funding_above_sma = funding > self.funding_sma[-1]
            price_inside_bands = price > self.lower[-1]
            
            if funding_above_sma:
                current_position.close()
                print(f"🌙💫 Funding normalized! Exiting at {price:.2f}")
            elif price_inside_bands:
                current_position.close()
                print(f"🚨🌙 Price returned to bands! Emergency exit at {price:.2f}")

# Moon Dev backtest execution 🌕
if __name__ == "__main__":
    data = load_data('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
    
    bt = Backtest(
        data,
        FundingBandBreach,
        cash=1_000_000,
        commission=.002,
        exclusive_orders=True
    )
    
    print("🌙✨ Starting Moon Dev Backtest... Hold for lunar orbit!")
    stats = bt.run()
    print("\n🌌🌕 BACKTEST COMPLETE