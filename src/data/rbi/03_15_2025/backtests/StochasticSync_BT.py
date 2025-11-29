```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class StochasticSync(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    swing_period = 20
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators using TA-Lib with self.I()
        # SMAs
        self.sma50 = self.I(talib.SMA, self.data.Close, timeperiod=50)
        self.sma200 = self.I(talib.SMA, self.data.Close, timeperiod=200)
        
        # Stochastic Oscillator
        stoch_k, stoch_d = talib.STOCH(self.data.High, self.data.Low, self.data.Close,
                                      fastk_period=14, slowk_period=3, slowd_period=3)
        self.stoch_k = self.I(lambda: stoch_k, name='%K')
        self.stoch_d = self.I(lambda: stoch_d, name='%D')
        
        # Swing High/Low
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period)
        
        print("🌙✨ Moon Dev Indicators Initialized! SMA50, SMA200, Stochastic, Swing Levels ready 🚀")

    def next(self):
        current_close = self.data.Close[-1]
        equity = self.equity
        
        # Entry Logic
        if not self.position:
            # Long Entry Conditions
            if (self.sma50[-1] > self.sma200[-1] and
                self.stoch_k[-1] > 20 and self.stoch_k[-2] <= 20 and
                crossover(self.stoch_k, self.stoch_d)):
                
                sl_price = self.swing_low[-1]
                risk_amount = equity * self.risk_per_trade
                risk_per_share = current_close - sl_price
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.buy(size=position_size, sl=sl_price)
                        print(f"🌙✨🚀 LONG ENTRY: {position_size} shares at {current_close:.2f} | SL: {sl_price:.2f} | Moon Power Activated! 🌕")
            
            # Short Entry Conditions
            elif (self.sma50[-1] < self.sma200[-1] and
                  self.stoch_k[-1] < 80 and self.stoch_k[-2] >= 80 and
                  crossover(self.stoch_d, self.stoch_k)):
                
                sl_price = self.swing_high[-1]
                risk_amount = equity * self.risk_per_trade
                risk_per_share = sl_price - current_close
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.sell(size=position_size, sl=sl_price)
                        print(f"🌙✨🚀 SHORT ENTRY: {position_size} shares at {current_close:.2f} | SL: {sl_price:.2f} | Lunar Gravity Engaged! 🌑")
        
        # Exit Logic
        else:
            if self.position.is_long:
                # Trend Reversal Exit
                if self.sma50[-1] < self.sma200[-1]:
                    self.position.close()
                    print(f"🌙🌕 LONG EXIT: SMA Death Cross Detected! Moon Radiation Shield Activated!")
                # Take Profit Exit
                elif self.stoch_k[-1] >= 80:
                    self.position.close()
                    print(f"🌙💎 LONG TP: Stochastic Overbought! Moon Diamond Hands Cash Out! 💰")
            
            elif self.position.is_short:
                # Trend Reversal Exit
                if self.sma50[-1] > self.sma200[-1]:
                    self.position.close()
                    print(f"🌙🌑 SHORT EXIT: SMA Golden Cross Detected! Lunar Retreat Protocol!")
                # Take Profit Exit
                elif self.stoch_k[-1] <= 20:
                    self.position.close()
                    print(f"🌙💎 SHORT TP: Stochastic Oversold! Moon Mining Profit Collected! ⛏️")

# Data Handling
data_path