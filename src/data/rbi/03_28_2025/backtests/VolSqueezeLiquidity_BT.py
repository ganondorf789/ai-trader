Here's the complete backtesting implementation for the VolSqueezeLiquidity strategy:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Data preprocessing moon-style 🌙
def load_data(data_path):
    data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    
    # Moon Dev magic column mapping ✨
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'funding_rate': 'Funding_Rate'  # Critical for strategy!
    })
    
    return data

class VolSqueezeLiquidity(Strategy):
    # Moon-tuned parameters 🌕
    bb_period = 20
    bb_dev = 2
    rsi_period = 14
    squeeze_threshold = 0.5  # 50% of average bandwidth
    volume_multiplier = 1.5  # 150% of average volume
    risk_per_trade = 0.02  # 2% risk per trade
    profit_multiplier = 1.5  # 1.5x BB width

    def init(self):
        # Moon-powered indicators 🌙💫
        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(talib.BBANDS, 
            self.data.Close, 
            timeperiod=self.bb_period,
            nbdevup=self.bb_dev,
            nbdevdn=self.bb_dev,
            matype=0
        )
        
        # RSI for overbought check
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period)
        
        # Volume analysis
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20)
        
        # Squeeze detection
        self.bb_width = self.I(lambda: self.bb_upper - self.bb_lower)
        self.bb_width_avg = self.I(talib.SMA, self.bb_width, 50)

    def next(self):
        # Moon Dev safety checks 🚀
        if len(self.data) < 50:  # Ensure enough data
            return
            
        # Current market conditions
        price = self.data.Close[-1]
        funding = self.data.Funding_Rate[-1]
        
        # Moon entry checklist 🌙✅
        squeeze_active = self.bb_width[-1] < (self.bb_width_avg[-1] * self.squeeze_threshold)
        funding_negative = funding < -0.01  # -0.01% funding rate
        price_breakout = price > self.bb_upper[-1]
        volume_surge = self.data.Volume[-1] > (self.volume_sma[-1] * self.volume_multiplier)
        rsi_safe = self.rsi[-1] < 70

        # Moon launch sequence! 🚀🌙
        if not self.position and all([
            squeeze_active,
            funding_negative,
            price_breakout,
            volume_surge,
            rsi_safe
        ]):
            # Calculate moon-sized position 🌕
            risk_amount = self.risk_per_trade * self.equity
            stop_loss = self.data.Low[-1]
            risk_per_unit = price - stop_loss
            
            if risk_per_unit > 0:
                position_size = int(round(risk_amount / risk_per_unit))
                take_profit = price + (self.bb_width[-1] * self.profit_multiplier)
                
                if position_size > 0:
                    self.buy(
                        size=position_size,
                        sl=stop_loss,
                        tp=take_profit
                    )
                    print(f"🚀🌙 MOON LAUNCH! Long {position_size} units")
                    print(f"   Entry: {price:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")

    def notify_trade(self, trade):
        if trade.is_closed:
            profit = trade.pl_pct
            if profit > 0:
                print(f"✨💰 MOON PROFIT! +{profit:.2%}")
            else:
                print(f"🔥🚨 COSMIC BURN! {profit:.2%}")

# Moon Dev