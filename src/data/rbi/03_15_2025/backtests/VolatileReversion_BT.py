Here's the complete backtesting implementation for the VolatileReversion strategy:

```python
# 🌙 Moon Dev's VolatileReversion Backtest 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Clean and prepare data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv', parse_dates=['datetime'])
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class VolatileReversion(Strategy):
    risk_per_trade = 0.01  # 1% of equity
    consecutive_losses = 0
    
    def init(self):
        # 🌙 Core Indicators
        self.ema200 = self.I(talib.EMA, self.data.Close, 200, name='EMA 200')
        self.rsi = self.I(talib.RSI, self.data.Close, 14, name='RSI 14')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR 14')
        
        # 🌟 Volatility Indicators
        self.vix = self.I(talib.SMA, self.data.Close, 20)  # Placeholder for VIX
        self.vix_sma = self.I(talib.SMA, self.vix, 20, name='VIX SMA 20')
        
        # 📈 Swing Detection
        self.price_lows = self.I(talib.MIN, self.data.Low, 5, name='Price Lows')
        self.rsi_lows = self.I(talib.MIN, self.rsi, 5, name='RSI Lows')
        self.volume_trend = self.I(talib.LINEARREG_SLOPE, self.data.Volume, 5, name='Volume Trend')

    def next(self):
        # 🌙 Moon Dev's Safety Checks
        if self.consecutive_losses >= 3:
            print("🚨🌙 3 Consecutive Losses - Trading Halted!")
            return
            
        # 🌟 Entry Conditions
        if not self.position and len(self.rsi) > 5:
            # Bullish Divergence Check
            div_condition = (
                self.price_lows[-1] < self.price_lows[-2] and 
                self.rsi_lows[-1] > self.rsi_lows[-2]
            )
            
            # Trend & Volume Check
            trend_condition = (
                self.data.Close[-1] > self.ema200[-1] and
                self.volume_trend[-1] < 0
            )
            
            # Volatility Check
            vix_condition = self.vix[-1] > self.vix_sma[-1]
            
            if div_condition and trend_condition and vix_condition:
                # 🌙 Risk Management Calculations
                atr = self.atr[-1]
                stop_loss = self.price_lows[-1] - 0.5*atr
                risk_amount = self.broker.equity * self.risk_per_trade
                risk_per_share = self.data.Close[-1] - stop_loss
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    self.buy(size=position_size, sl=stop_loss, 
                            tp=self.data.Close[-1] + 1.5*atr)
                    
                    print(f"🚀🌙 LONG Entry: {self.data.Close[-1]:.2f}")
                    print(f"   SL: {stop_loss:.2f} | TP: {self.data.Close[-1]+1.5*atr:.2f}")

        # 🌟 Exit Conditions
        if self.position:
            # Check trend reversal
            if self.data.Close[-1] < self.ema200[-1]:
                self.position.close()
                print(f"📉🌙 Trend Reversal Exit: {self.data.Close[-1]:.2f}")
            
            # Check volatility drop
            elif self.vix[-1] < self.vix_sma[-1]:
                self.position.close()
                print(f"🌪️🌙 Volatility Drop Exit: {self.data.Close[-1]:.2f}")

    def notify_trade(self, trade):
        if trade.is_cl