I'll help debug the code while maintaining the strategy logic and adding Moon Dev themed debug prints. Here's the fixed version:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from datetime import datetime, timedelta
from typing import List

# Data preprocessing
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class GammaSurgeMomentum(Strategy):
    risk_pct = 0.02
    tp_pct = 0.10
    sl_pct = 0.02
    exit_days = 3
    
    def init(self):
        self.expiration_dates = self._get_expiration_times()
        
        # Core indicators
        self.max_high = self.I(talib.MAX, self.data.High, 8)
        self.vol_sma = self.I(talib.SMA, self.data.Volume, 5)
        self.macd_line = self.I(self._calc_macd_line)
        self.macd_signal = self.I(self._calc_macd_signal)
        
        print("🌙✨ MOON DEV INIT: Strategy initialized with cosmic indicators!")
        
    def _calc_macd_line(self):
        macd, _, _ = talib.MACD(self.data.Close, 12, 26, 9)
        return macd
    
    def _calc_macd_signal(self):
        _, signal, _ = talib.MACD(self.data.Close, 12, 26, 9)
        return signal
    
    def _get_expiration_times(self) -> List[datetime]:
        expirations = []
        for month in range(1, 13):
            date = pd.date_range(
                start=f'2023-{month}-01', 
                end=f'2023-{month}-28', 
                freq='WOM-3FRI'
            )[0].to_pydatetime().replace(hour=21, minute=0)
            expirations.append(date)
        print(f"🌕📅 MOON DEV EXPIRATIONS: {len(expirations)} expiration dates calculated")
        return expirations
    
    def _in_trading_window(self):
        current_time = self.data.index[-1]
        for exp_time in self.expiration_dates:
            if exp_time <= current_time <= (exp_time + timedelta(hours=2)):
                return True
        return False
    
    def next(self):
        if not self._in_trading_window() or self.position:
            return
            
        # Entry conditions check
        price_break = self.data.Close[-1] > self.max_high[-1]
        volume_boost = self.data.Volume[-1] > self.vol_sma[-1]
        macd_bullish = (self.macd_line[-2] < self.macd_signal[-2] and 
                       self.macd_line[-1] > self.macd_signal[-1])
        
        if all([price_break, volume_boost, macd_bullish]):
            # Position sizing
            equity = self.equity
            entry_price = self.data.Close[-1]
            sl_price = entry_price * (1 - self.sl_pct)
            tp_price = entry_price * (1 + self.tp_pct)
            
            risk_amount = equity * self.risk_pct
            position_size = int(round(risk_amount / (entry_price - sl_price)))
            
            if position_size > 0:
                self.buy(
                    size=position_size,
                    sl=sl_price,
                    tp=tp_price,
                    tag=f"🌙 ENTRY: {entry_price:.2f} 🚀"
                )
                print(f"🌕✨ MOON DEV TRADE: {position_size} units @ {entry_price:.2f}")
                print(f"🌙🛡️ MOON DEV RISK: SL={sl_price:.2f}, TP={tp_price:.2f}")
                
        # Time-based exits
        for trade in self.trades:
            if trade.is_open and (self.data.index[-1] >= trade.entry_time + pd.Timedelta(days=self.exit_days)):
                self.position.close