```python
# 🌙 Moon Dev's VoltaicCompression Backtest Implementation
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# 🌌 Data Preparation Magic
def prepare_data(path):
    data = pd.read_csv(path)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Remove unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    # Map to proper column names
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    # Convert and set index
    data['DateTime'] = pd.to_datetime(data['datetime'])
    return data.set_index('DateTime')

# 🚀 VoltaicCompression Strategy Class
class VoltaicCompression(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌗 Calculate Indicators using TA-Lib
        self.middle_band = self.I(talib.SMA, self.data.Close, 20, name='SMA20')
        self.std_dev = self.I(talib.STDDEV, self.data.Close, 20, 1, name='STDDEV')
        self.upper_band = self.I(lambda sma, std: sma + 2*std, self.middle_band, self.std_dev, name='UPPER')
        self.lower_band = self.I(lambda sma, std: sma - 2*std, self.middle_band, self.std_dev, name='LOWER')
        self.bb_width = self.I(lambda u,l,m: (u-l)/m, self.upper_band, self.lower_band, self.middle_band, name='BBWIDTH')
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, 14, name='ADX14')
        
    def next(self):
        # 🌑 Skip early bars without indicator data
        if len(self.data) < 20:
            return
            
        # 🌕 Current indicator values
        bb_width = self.bb_width[-1]
        adx_value = self.adx[-1]
        price_close = self.data.Close[-1]
        
        # 🌙 Debug Prints
        print(f"🌙 MOON DEV DEBUG || BB Width: {bb_width:.4f} | ADX: {adx_value:.2f} | Close: {price_close:.2f}")
        
        # 🛑 Skip trades during news events (placeholder)
        
        # 💹 Entry Logic
        if not self.position:
            # Volatility compression + weak trend
            if bb_width < 0.5 and adx_value < 25:
                # Long Entry
                if price_close > self.upper_band[-1]:
                    self.enter_long()
                # Short Entry
                elif price_close < self.lower_band[-1]:
                    self.enter_short()
        else:
            # 💸 Exit Logic
            if self.position.is_long:
                if price_close < self.middle_band[-1]:
                    self.position.close()
                    print(f"🌙✨ PROFIT MOONSHOT! Closed LONG at {price_close:.2f}")
            elif self.position.is_short:
                if price_close > self.middle_band[-1]:
                    self.position.close()
                    print(f"🌙✨ SHORT CIRCUIT! Closed SHORT at {price_close:.2f}")
    
    def enter_long(self):
        entry_price = self.data.Open[-1]
        stop_loss = self.lower_band[-1]
        risk_amount = self.equity * self.risk_percent
        risk_per_share = entry_price - stop_loss
        
        if risk_per_share <= 0:
            print("🌑 MOON ALERT! Negative risk detected - aborting launch")
            return
            
        position_size = int(round(risk_amount / risk_per_share))
        if position_size == 0:
            return
            
        take_profit = entry_price + 2*(entry_price - stop_loss)
        self.buy(size=position_size, sl=stop_loss, tp=take_profit)
        print(f"🚀🌕 MOONSHOT LAUNCH! LONG {position_size} @ {entry_price:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
    
    def enter_short(self):
        entry_price = self.data.Open[-1]
        stop_loss = self.upper_band