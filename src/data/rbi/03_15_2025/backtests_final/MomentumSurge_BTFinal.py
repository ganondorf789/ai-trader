The code looks mostly correct, but I'll make a few technical improvements and add some Moon Dev themed debug prints for better clarity. Here's the fixed version:

```python
# 🌙 Moon Dev Backtest Implementation for MomentumSurge Strategy
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class MomentumSurge(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌙 Calculate indicators using TA-Lib with self.I()
        self.sma50 = self.I(talib.SMA, self.data.Close, timeperiod=50, name='SMA50')
        self.sma200 = self.I(talib.SMA, self.data.Close, timeperiod=200, name='SMA200')
        self.rsi14 = self.I(talib.RSI, self.data.Close, timeperiod=14, name='RSI14')
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR14')
        self.volume_roc = self.I(talib.ROC, self.data.Volume, timeperiod=1, name='VolumeROC')
        
        print("🌙✨ Moon Dev Indicators Initialized! Ready for lunar trading! 🌕")

    def next(self):
        price = self.data.Close[-1]
        
        # 🚀 Long Entry Conditions
        if not self.position:
            # Golden Cross Check (manual crossover detection)
            golden_cross = (self.sma50[-2] < self.sma200[-2]) and (self.sma50[-1] > self.sma200[-1])
            # RSI Oversold Check
            rsi_condition = self.rsi14[-1] < 30
            # Volume Spike Check (10% increase)
            volume_condition = self.volume_roc[-1] > 10

            if golden_cross and rsi_condition and volume_condition:
                # 🌙 Risk Management Calculations
                atr_value = self.atr14[-1]
                stop_loss = price - 2 * atr_value
                risk_per_share = price - stop_loss
                
                if risk_per_share > 0:
                    position_size = (self.broker.equity * self.risk_per_trade) / risk_per_share
                    position_size = int(round(position_size))  # Ensure whole number of units
                    
                    if position_size > 0:
                        self.buy(size=position_size, sl=stop_loss)
                        print(f"🚀🌙 MOON DEV ENTRY 🚀 | Price: {price:.2f} | Size: {position_size} | SL: {stop_loss:.2f} | Lunar Launch Detected! 🌖")

        # ✨ Exit Conditions
        elif self.position:
            # Death Cross Check (manual crossover detection)
            death_cross = (self.sma200[-2] < self.sma50[-2]) and (self.sma200[-1] > self.sma50[-1])
            # RSI Overbought Check
            rsi_exit = self.rsi14[-1] >= 70
            
            if death_cross or rsi_exit:
                self.position.close()
                reason = "DEATH CROSS" if death_cross else "RSI OVERBOUGHT"
                print(f"🌙✨ MOON DEV EXIT ✨ | Price: {price:.2f} | Reason: {reason} | Returning to lunar orbit 🌘")

# 🌙 Data Preparation
try:
    data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

    # 🪐 Ensure proper column mapping
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })

    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    
    print("🌙✨ Data successfully loaded and formatted! Ready for lunar analysis! 🌕")
    
except Exception as e:
    print(f"🌙⚠️ Lunar Data Error: {str(e)}")
    raise

# 🚀 Run Backtest
try:
    bt = Backtest(data, MomentumSurge, cash=1_000_000, exclusive_orders=True)
    stats = bt.run()
    
    # 🌙 Print Full