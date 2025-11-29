```python
# 🌙 Moon Dev's Volumetric EMA Surge Backtest 🌙
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib
import numpy as np

# 🚀 Data Preparation Rocket Launch 🚀
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# 🌌 Cosmic Data Cleaning 🌌
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

class VolumetricEMASurge(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌕
    
    def init(self):
        # ✨ Indicator Constellation Setup ✨
        self.ema200 = self.I(talib.EMA, self.data.Close, 200, name='EMA200')
        self.volume_avg20 = self.I(talib.SMA, self.data.Volume, 20, name='Vol20')
        self.rsi14 = self.I(talib.RSI, self.data.Close, 14, name='RSI14')
        
        # 🌠 Moon Dev Debug Initialization 🌠
        print("🌙 Lunar Strategy Activated - 200 EMA | 2x Volume Surge | RSI14 Guard")

    def next(self):
        # 🌔 Orbital Position Tracking 🌔
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        
        # 🌓 Cosmic Debug Prints (Every 100 Bars) 🌓
        if len(self.data) % 100 == 0:
            print(f"\n🌕 Moon Phase Update 🌕\n"
                  f"| Close: {current_close:.2f}\n"
                  f"| EMA200: {self.ema200[-1]:.2f}\n"
                  f"| Volume: {current_volume:.2f} vs {self.volume_avg20[-1]:.2f}\n"
                  f"| RSI14: {self.rsi14[-1]:.2f}\n"
                  "✨"*30)

        # 🚀 Launch Entry Sequence 🚀
        if not self.position:
            if (not np.isnan(self.ema200[-1]) and 
                current_close > self.ema200[-1] and 
                current_volume > 2 * self.volume_avg20[-1]):
                
                # 💰 Risk Management Engine 💰
                entry_price = current_close
                stop_loss = entry_price * 0.98  # 2% Stellar Protection
                risk_amount = self.broker.get_value() * self.risk_percent
                position_size = int(round(risk_amount / (entry_price - stop_loss)))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_loss)
                    print(f"\n🚀🌙 BLASTOFF! Long {position_size} @ {entry_price:.2f}\n"
                          f"| Stop Loss: {stop_loss:.2f}\n"
                          f"| Portfolio Risk: {self.risk_percent*100}%\n"
                          "✨"*30)

        # 🌑 Exit Protocol Activation 🌑
        else:
            exit_signal = False
            # RSI Reversal Detection
            if len(self.rsi14) > 1 and self.rsi14[-1] < 70 and self.rsi14[-2] >= 70:
                print(f"🌑 RSI Fading Star Signal @ {self.rsi14[-1]:.2f}")
                exit_signal = True
                
            # EMA Shield Breach
            if current_close < self.ema200[-1]:
                print(f"🌑 EMA200 Cosmic Barrier Broken @ {current_close:.2f}")
                exit_signal = True
                
            if exit_signal:
                self.position.close()
                print(f"\n🌑🌙 RETURN TO BASE!\n"
                      f"| Exit Price: {current_close:.2f}\n"
                      f"| Profit: {self.position.pl_pct:.2%}\n"
                      "✨"*30)

# 🌕 Backtest Launch Sequence 🌕
bt = Backtest(data, VolumetricEMASur