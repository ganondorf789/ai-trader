Here's the complete backtesting implementation for VoltaicReversion strategy:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv', parse_dates=['datetime'])
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}).set_index('datetime')

class VoltaicReversion(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    
    def init(self):
        # Moon Dev Indicators 🌙
        self.stoch_k, self.stoch_d = self.I(talib.STOCH, 
            self.data.High, self.data.Low, self.data.Close,
            fastk_period=14, slowk_period=3, slowk_matype=0,
            slowd_period=3, slowd_matype=0)
        
        self.sma200 = self.I(talib.SMA, self.data.Close, 200)
        self.atr = self.I(talib.ATR, 
            self.data.High, self.data.Low, self.data.Close, 14)
        
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(talib.BBANDS,
            self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    def next(self):
        # Moon Dev Entry Logic 🌟
        if not self.position:
            # Long constellation alignment ✨
            long_conditions = (
                self.data.Close[-1] > self.sma200[-1],
                crossover(self.stoch_k, self.stoch_d),
                self.stoch_k[-2] < 20 and self.stoch_d[-2] < 20,
                self.data.Low[-1] <= self.bb_lower[-1],
                self.atr[-1] < self.atr[-4]  # Decreasing volatility
            )
            
            # Short constellation alignment 🌌
            short_conditions = (
                self.data.Close[-1] < self.sma200[-1],
                crossunder(self.stoch_k, self.stoch_d),
                self.stoch_k[-2] > 80 and self.stoch_d[-2] > 80,
                self.data.High[-1] >= self.bb_upper[-1],
                self.atr[-1] < self.atr[-4]
            )

            if all(long_conditions):
                self.enter_long()
                
            elif all(short_conditions):
                self.enter_short()
        
        # Moon Dev Exit Management 🌙
        for trade in self.trades:
            # Time-based exit (5 days in 15m intervals)
            if len(self.data) - trade.tag['entry_bar'] >= 480:
                print(f"🌙⏰ Time exit after 5 days! Closing {trade.tag['type']}")
                trade.close()
                continue
            
            # Cosmic trailing stops 🌠
            self.update_trailing_stop(trade)

            # Stellar exit conditions 🌟
            if trade.tag['type'] == 'long' and (self.stoch_k[-1] >= 80 or self.stoch_d[-1] >= 80):
                print(f"🌙📈 Overbought supernova! Closing long")
                trade.close()
            elif trade.tag['type'] == 'short' and (self.stoch_k[-1] <= 20 or self.stoch_d[-1] <= 20):
                print(f"🌙📉 Oversold black hole! Closing short")
                trade.close()

    def enter_long(self):
        # Galactic position sizing 🌌
        atr_value = self.atr[-1]
        risk_amount = self.equity * self.risk_pct
        position_size = int(round(risk_amount / atr_value)) or 1
        
        # Stardust entry parameters 🌠
        entry_price = self.data.Close[-1]
        sl = entry_price - atr_value
        tp = entry_price + 2 * atr_value
        
        self.buy(
            size=position_size,
            sl=sl,
            tp=tp,
            tag={
                'type': 'long',
                'entry_price': entry_price,
                'atr_entry': atr_value,
                '