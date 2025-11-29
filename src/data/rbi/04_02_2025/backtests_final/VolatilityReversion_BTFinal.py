Here's the fully debugged backtest code with Moon Dev themed improvements while preserving the original strategy logic:

```python
# 🌙 Moon Dev's Volatility Reversion Backtest 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class VolatilityReversion(Strategy):
    risk_per_trade = 0.01  # 1% of equity per trade
    
    def init(self):
        # 🌌 Cosmic Indicator Setup 🌌
        close = self.data.Close
        
        # Bollinger Bands with Bandwidth Calculation
        self.upper, self.middle, self.lower = self.I(talib.BBANDS, close, 
                                                    timeperiod=20, nbdevup=2, nbdevdn=2,
                                                    matype=0)
        self.bandwidth = (self.upper - self.lower) / self.middle
        self.avg_bandwidth = self.I(talib.SMA, self.bandwidth, 5)
        
        # 📡 Momentum Indicators
        self.rsi = self.I(talib.RSI, close, 14)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, close, 14)
        
        # 🪐 Open Interest Tracking
        self.oi_high = self.I(talib.MAX, self.data.open_interest, 20)
        
        # 🌑 Swing Lows for Stop Loss
        self.swing_low = self.I(talib.MIN, self.data.Low, 20)
        
        print("✨ Moon Base Indicators Initialized! ✨")

    def next(self):
        # 🌠 Lunar Cycle Conditions 🌠
        current_funding = self.data.funding_rate[-1]
        current_oi = self.data.open_interest[-1]
        oi_peak = self.oi_high[-1]
        
        # Entry Signals
        funding_cond = current_funding < -0.001
        oi_cond = (oi_peak - current_oi)/oi_peak > 0.1 if oi_peak != 0 else False
        vol_cond = self.bandwidth[-1] > 2*self.avg_bandwidth[-1]
        
        if not self.position and funding_cond and oi_cond and vol_cond:
            # 🚀 Rocket Position Sizing 🚀
            atr = self.atr[-1] if self.atr[-1] != 0 else 1
            risk_amount = self.equity * self.risk_per_trade
            position_size = risk_amount / atr
            position_size = int(round(position_size))  # Ensure whole number units
            
            if position_size > 0:
                sl_price = self.swing_low[-1]
                tp_price = self.data.Close[-1] + 2*atr
                
                self.buy(size=position_size, 
                        sl=sl_price,
                        tp=tp_price,
                        tag="🌕 Moon Shot")
                print(f"🚀 LIFTOFF! Entry: {self.data.Close[-1]:.2f} | Size: {position_size} | SL: {sl_price:.2f}")

        # 🌘 Exit Conditions 🌘
        if self.position:
            # Funding Rate Normalization
            if current_funding >= -0.0002:
                self.position.close()
                print(f"🌓 Funding Stabilized | Exit: {self.data.Close[-1]:.2f}")
            
            # RSI Divergence Check
            elif self.rsi[-1] > 70 and self.rsi[-1] > self.rsi[-2]:
                self.position.close()
                print(f"📉 RSI Peak Detected | Exit: {self.data.Close[-1]:.2f}")

# 🛰 Data Loading & Preparation 🛰
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# 🧹 Data Cleansing Ritual 🧹
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

# 🌌 Column Alchemy 🌌
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'datetime': 'Date',
    'open_interest': 'open_interest',
    'funding_rate': 'funding_rate'
}, inplace=True)

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

print("🌕 Moon Data