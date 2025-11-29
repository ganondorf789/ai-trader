```python
import pandas as pd
import talib
from backtesting import Strategy, Backtest
import numpy as np

# Data preparation
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean and format data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolumetricSqueeze(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Core indicators
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume, name='OBV')
        self.obv_20_high = self.I(talib.MAX, self.obv, 20, name='OBV_20_HIGH')
        self.obv_20_low = self.I(talib.MIN, self.obv, 20, name='OBV_20_LOW')
        
        # Keltner Channel components
        self.ema20 = self.I(talib.EMA, self.data.Close, 20, name='EMA20')
        self.atr20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 20, name='ATR20')
        self.kc_width = self.I(lambda arr: arr*4, self.atr20, name='KC_WIDTH')
        self.avg_kc_width = self.I(talib.SMA, self.kc_width, 20, name='AVG_KC_WIDTH')
        
    def next(self):
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        
        # Calculate dynamic Keltner levels
        ema = self.ema20[-1]
        atr = self.atr20[-1]
        upper_kc = ema + 2*atr
        lower_kc = ema - 2*atr
        
        # Strategy conditions
        obv_high_condition = self.obv[-1] >= self.obv_20_high[-1]
        squeeze_condition = self.kc_width[-1] < self.avg_kc_width[-1]
        
        # Moon Dev debug prints 🌙
        if obv_high_condition:
            print(f"🌕 OBV 20-day HIGH! Current: {self.obv[-1]:.2f}, High: {self.obv_20_high[-1]:.2f}")
        if squeeze_condition:
            print(f"✨ KC SQUEEZE! Width: {self.kc_width[-1]:.2f} < Avg: {self.avg_kc_width[-1]:.2f}")
        
        # Entry logic
        if not self.position:
            if obv_high_condition and squeeze_condition:
                # Long entry
                if current_close > upper_kc:
                    risk_amount = self.equity * self.risk_per_trade
                    sl_price = lower_kc
                    risk_per_share = current_close - sl_price
                    
                    if risk_per_share > 0:
                        position_size = int(round(risk_amount / risk_per_share))
                        self.buy(size=position_size, sl=sl_price)
                        print(f"🚀 LONG ENTRY: {current_close:.2f}, Size: {position_size}, SL: {sl_price:.2f}")
                
                # Short entry
                elif current_close < lower_kc:
                    risk_amount = self.equity * self.risk_per_trade
                    sl_price = upper_kc
                    risk_per_share = sl_price - current_close
                    
                    if risk_per_share > 0:
                        position_size = int(round(risk_amount / risk_per_share))
                        self.sell(size=position_size, sl=sl_price)
                        print(f"🌑 SHORT ENTRY: {current_close:.2f}, Size: {position_size}, SL: {sl_price:.2f}")
        
        # Exit logic
        else:
            if self.position.is_long:
                if current_close < ema:
                    self.position.close()
                    print(f"🛑 LONG