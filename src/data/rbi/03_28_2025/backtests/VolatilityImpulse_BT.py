```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class VolatilityImpulse(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Calculate indicators with proper self.I() wrapping
        self.cmo = self.I(talib.CMO, self.data.Close, 20, name='CMO_20')
        self.cmo_std = self.I(talib.STDDEV, self.cmo, 20, name='CMO_STD_20')
        self.threshold = self.I(lambda x: x * 1.5, self.cmo_std, name='Dynamic_Threshold')
        self.ema13 = self.I(talib.EMA, self.data.Close, 13, name='EMA_13')
        self.macd_hist = self.I(lambda close: talib.MACD(close, 12, 26, 9)[2], self.data.Close, name='MACD_Hist')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 20, name='ATR_20')

    def next(self):
        price = self.data.Close[-1]
        
        if not self.position:
            # Entry logic
            if len(self.cmo) < 2 or len(self.threshold) < 2:
                return
            
            # Get indicator values
            current_threshold = self.threshold[-1]
            cmo_above = crossover(self.cmo, self.threshold)[-1]
            
            # Elder's Impulse check
            ema_current = self.ema13[-1]
            ema_prev = self.ema13[-2] if len(self.ema13) >=2 else ema_current
            macd_current = self.macd_hist[-1]
            elder_bullish = (ema_current > ema_prev) and (macd_current > 0)
            
            if cmo_above and elder_bullish and (current_threshold >= 10):
                # Risk management calculations
                atr_value = self.atr[-1]
                stop_loss = price - atr_value
                risk_amount = self.equity * self.risk_percent
                risk_per_share = price - stop_loss
                
                if risk_per_share <= 0:
                    return
                
                position_size = int(round(risk_amount / risk_per_share))
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_loss, tag='🌙 Moon Entry')
                    print(f"🚀 MOON DEV LONG SIGNAL 🚀\nEntry: {price:.2f} | Size: {position_size} | SL: {stop_loss:.2f} | Equity: {self.equity:.2f}")
        else:
            # Exit conditions
            ema_current = self.ema13[-1]
            ema_prev = self.ema13[-2] if len(self.ema13) >=2 else ema_current
            macd_current = self.macd_hist[-1]
            
            # Elder's bearish check
            elder_bearish = (ema_current < ema_prev) and (macd_current < 0)
            # CMO below threshold check
            cmo_below = self.cmo[-1] < self.threshold[-1]
            
            if elder_bearish:
                self.position.close()
                print(f"🌑 ELDER EXIT SIGNAL 🌑\nPrice: {price:.2f} | Profit: {self.position.pl:.2f}")
            elif cmo_below:
                self.position.close()
                print(f"🌘 CMO EXIT SIGNAL 🌘\nPrice: {price:.2f} | Profit: {self.position.pl:.2f}")

if __name__ == '__main__':
    # Load and preprocess data
    data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv', parse_dates=['datetime'])
    
    # Clean columns
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    data.set_index('datetime', inplace=True)

    # Run backtest
    bt = Backtest(data, VolatilityImpulse, cash=1_