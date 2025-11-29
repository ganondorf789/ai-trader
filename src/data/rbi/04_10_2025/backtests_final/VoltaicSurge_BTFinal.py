# 🌙 MOON DEV DEBUGGED BACKTESTING SCRIPT FOR VOLTAIC SURGE STRATEGY 🚀✨

# ===== REQUIRED IMPORTS =====
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

# ===== DATA PREPARATION =====
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"

# Load and clean data with Moon Dev standards 🌙
try:
    data = pd.read_csv(data_path)
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    
    # Format columns to match backtesting.py requirements
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    print("🌙✨ MOON DEV DATA: Data loaded and formatted successfully!")
except Exception as e:
    print(f"🌑 MOON DEV ERROR: Data loading failed - {str(e)}")
    raise

# ===== VOLTAIC SURGE STRATEGY CLASS =====
class VoltaicSurge(Strategy):
    # Strategy parameters 🌌
    atr_short_period = 20        # 15m ATR (20 periods = 5hrs)
    atr_daily_period = 14*96     # 14 days in 15m intervals (96 bars/day)
    volume_ma_period = 20
    risk_pct = 0.01              # 1% risk per trade
    
    def init(self):
        # 🌙 Core indicators using TA-Lib
        try:
            self.atr_short = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_short_period)
            self.atr_daily_avg = self.I(talib.SMA, self.atr_short, timeperiod=self.atr_daily_period)
            self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_ma_period)
            print("🌙✨ MOON DEV INIT: Strategy indicators initialized successfully! 🚀")
        except Exception as e:
            print(f"🌑 MOON DEV ERROR: Indicator initialization failed - {str(e)}")
            raise

    def next(self):
        # Current market conditions 🌊
        current_atr = self.atr_short[-1]
        daily_atr_avg = self.atr_daily_avg[-1]
        current_volume = self.data.Volume[-1]
        volume_threshold = self.volume_ma[-1]

        # 🌙 Entry Logic: Volatility Breakout with Volume Confirmation
        if not self.position:
            if (current_atr > 2 * daily_atr_avg) and (current_volume > volume_threshold):
                entry_price = self.data.Open[-1]
                sl = entry_price - daily_atr_avg
                tp = entry_price + 2 * daily_atr_avg
                
                # Moon Dev Risk Management 🌙
                risk_amount = self.risk_pct * self.equity
                risk_per_share = entry_price - sl
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    try:
                        self.buy(size=position_size, sl=sl, tp=tp)
                        print(f"🌙🚀 MOON DEV ENTRY: LONG {position_size} units @ {entry_price:.2f}")
                        print(f"   🌑 SL: {sl:.2f} | 🌕 TP: {tp:.2f} | Risk: {self.risk_pct*100}% of equity")
                    except Exception as e:
                        print(f"🌑 MOON DEV ERROR: Trade execution failed - {str(e)}")

# ===== BACKTEST EXECUTION =====
try:
    bt = Backtest(data, VoltaicSurge, cash=1_000_000, commission=.002)
    stats = bt.run()
    print("\n🌕🌖🌗🌘🌑 MOON DEV FINAL STATS 🌑🌒🌓🌔🌕")
    print(stats)
    print("\n🌙 STRATEGY DETAILS:")
    print(stats._strategy)
    print("\n🚀 MOON DEV BACKTEST COMPLETE! TO