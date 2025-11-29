```python
import pandas as pd
import numpy as np
import talib
from backtesting import Strategy, Backtest

# Clean and prepare data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Helper function for Bollinger Bands
def bb_bands(close, timeperiod=20, nbdevup=2, nbdevdn=2):
    upper, middle, lower = talib.BBANDS(close, timeperiod=timeperiod, 
                                      nbdevup=nbdevup, nbdevdn=nbdevdn)
    return upper, middle, lower

class VolatilityBreakout(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Core indicators 🌙
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(bb_bands, self.data.Close, 20, 2, 2)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20)
        self.rsi = self.I(talib.RSI, self.data.Close, 14)
        self.swing_low = self.I(talib.MIN, self.data.Low, 5)
        self.rsi_swing_low = self.I(talib.MIN, self.rsi, 5)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
    def next(self):
        # Moon Dev Debug Prints ✨
        if len(self.data) % 100 == 0:
            print(f"🌕 MOON DEV UPDATE 🌙 | Bar: {len(self.data)} | Price: {self.data.Close[-1]:.2f} | Equity: {self.equity:,.2f}")
        
        if not self.position:
            # Entry Conditions Check 🚀
            price_breakout = self.data.Close[-1] > self.bb_upper[-1]
            volume_trigger = self.data.Volume[-1] > self.volume_sma[-1]
            
            # Bullish Divergence Check
            if len(self.swing_low) > 2 and len(self.rsi_swing_low) > 2:
                div_cond1 = self.swing_low[-1] > self.swing_low[-2]
                div_cond2 = self.rsi_swing_low[-1] > self.rsi_swing_low[-2]
                divergence = div_cond1 and div_cond2
            else:
                divergence = False
                
            if price_breakout and volume_trigger and divergence:
                # Risk Management Calculation 🌗
                atr_value = self.atr[-1]
                if atr_value <= 0: return
                
                risk_amount = self.broker.equity * self.risk_percent
                position_size = int(round(risk_amount / atr_value))
                position_size = max(position_size, 1)  # Ensure minimum 1 unit
                
                # Entry Execution 🌟
                entry_price = self.data.Close[-1]
                stop_loss = entry_price - atr_value * 1
                take_profit = entry_price + atr_value * 2
                
                self.buy(size=position_size, sl=stop_loss, tp=take_profit)
                print(f"\n🚀 MOONSHOT ALERT 🌕 | ENTRY: {entry_price:.2f} | SIZE: {position_size} | RISK: {atr_value:.2f} | TP: {take_profit:.2f} | SL: {stop_loss:.2f}")
        else:
            # Exit Conditions 🌑
            if self.data.Close[-1] < self.bb_middle[-1]:
                self.position.close()
                print(f"\n🌑 LUNAR ECLIPSE EXIT 🌘 | CLOSE: {self.data.Close[-1]:.2f} | REASON: Middle Band Break")

# Launch Backtest 🌕✨
bt = Backtest(data, VolatilityBreakout, cash=1_000_000, commission=0