Here's the fixed code with all backtesting.lib imports removed and proper indicator implementations:

```python
# 🌙 Moon Dev's VoltaicCrush Backtest 🌙
import pandas as pd
import talib
import numpy as np
from backtesting import Strategy, Backtest

# Data preparation 🌐
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and prepare columns 🌟
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

class VoltaicCrush(Strategy):
    # Strategy parameters ⚙️
    z_period = 252 * 4 * 24  # 1-year equivalent in 15m intervals
    vvix_sma_period = 20
    bb_period = 10
    z_threshold = 2.0
    risk_pct = 0.01
    max_bars_held = 96 * 5  # 5 days in 15m bars

    def init(self):
        # Calculate indicators using TA-Lib wrapped in self.I() 🌗
        vix = self.data.df['vix']
        
        # VIX Z-Score calculation
        def z_score(series, period):
            sma = talib.SMA(series, timeperiod=period)
            std = talib.STDDEV(series, timeperiod=period)
            return (series - sma) / std
        self.vix_z = self.I(z_score, vix, self.z_period, name='VIX_Z')
        
        # VVIX SMA
        vvix = self.data.df['vvix']
        self.vvix_sma = self.I(talib.SMA, vvix, self.vvix_sma_period, name='VVIX_SMA')
        
        # Bollinger Band Midpoint
        self.bb_mid = self.I(talib.SMA, vix, self.bb_period, name='BB_MID')
        
        # Recent High for stop loss
        self.recent_high = self.I(talib.MAX, self.data.High, 20, name='RECENT_HIGH')
        
        self.entry_bar = 0

    def next(self):
        # Moon Dev progress tracker 🌙
        if len(self.data) % 100 == 0:
            print(f"🌙 Processing Bar {len(self.data)} - Equity: ${self.equity:,.2f}")
            
        if self.position:
            # Exit conditions ✨
            current_vix = self.data.df['vix'][-1]
            current_bb_mid = self.bb_mid[-1]
            
            if current_vix < current_bb_mid:
                self.position.close()
                print(f"✨ Moon Exit! VIX {current_vix:.2f} < BB Mid {current_bb_mid:.2f}")
                return
                
            if len(self.data) - self.entry_bar >= self.max_bars_held:
                self.position.close()
                print(f"⏰ Time Exit! Bars held: {len(self.data) - self.entry_bar}")
        else:
            # Entry conditions 🚀
            current_z = self.vix_z[-1]
            current_vvix = self.data.df['vvix'][-1]
            current_vvix_sma = self.vvix_sma[-1]
            
            if current_z > self.z_threshold and current_vvix < current_vvix_sma:
                # Risk management 💰
                risk_amount = self.equity * self.risk_pct
                entry_price = self.data.Close[-1]
                stop_price = self.recent_high[-1]
                risk_per_share = stop_price - entry_price
                
                if risk_per_share <= 0:
                    print("⚠️ Moon Alert! Invalid risk calculation")
                    return
                
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    self.sell(size=position_size, sl=stop_price)
                    self.entry_bar = len(self.data)
                    print(f"🌙🚀 SHORT ACTIVATED! Size: {position_size} | Entry: {entry_price:.2f} | Stop: {stop_price:.