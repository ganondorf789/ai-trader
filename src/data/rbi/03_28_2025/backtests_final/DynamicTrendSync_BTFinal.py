Here's the debugged backtest code with Moon Dev themed improvements while maintaining the original strategy logic:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Clean and prepare data with Moon Dev precision 🌙
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
try:
    data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    print("🌙✨ Data loaded successfully! First timestamp:", data.index[0])
except Exception as e:
    print(f"🌙💥 Data loading error: {str(e)}")
    raise

class DynamicTrendSync(Strategy):
    risk_pct = 0.01  # 1% risk per trade (proper fraction format)
    atr_multiplier = 2  # Whole number for ATR multiple
    
    def init(self):
        # Moon Dev Indicator Initialization ✨
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=50, name='EMA50 🌙')
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=200, name='EMA200 🌕')
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ADX 📈')
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14, name='RSI 📉')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR ⚡')
        print("🌙✨ Indicators initialized successfully!")

    def next(self):
        price = self.data.Close[-1]
        
        # Moon Dev Debug Dashboard ✨
        debug_info = (
            f"🌙 {self.data.index[-1]} | "
            f"Price: {price:.2f} | "
            f"EMA50: {self.ema50[-1]:.2f} | "
            f"EMA200: {self.ema200[-1]:.2f} | "
            f"ADX: {self.adx[-1]:.1f} | "
            f"RSI: {self.rsi[-1]:.1f}"
        )
        # Uncomment for detailed debug info:
        # print(debug_info)
        
        if not self.position:
            # Entry Logic 🌟
            bullish_cross = (self.ema50[-2] < self.ema200[-2] and self.ema50[-1] > self.ema200[-1])
            if bullish_cross and self.adx[-1] > 25 and self.adx[-1] > self.adx[-2]:
                risk_amount = self.equity * self.risk_pct
                atr_value = self.atr[-1]
                est_entry = self.data.Close[-1]  # Estimate entry at current close
                sl_price = est_entry - (self.atr_multiplier * atr_value)
                
                if sl_price <= 0:
                    print(f"🌙💥 Invalid SL: {sl_price:.2f} - Trade Canceled")
                    return
                
                risk_per_share = est_entry - sl_price
                if risk_per_share <= 0:
                    print(f"🌙💥 Risk/Share: {risk_per_share:.2f} - Trade Canceled")
                    return
                
                position_size = int(round(risk_amount / risk_per_share))  # Proper rounding to whole units
                
                if position_size > 0:
                    self.buy(size=position_size, sl=sl_price)
                    print(f"🚀🌙 MOON DEV LONG ENTRY ✨ | "
                          f"Size: {position_size} units | "
                          f"Entry: {est_entry:.2f} | "
                          f"SL: {sl_price:.2f} | "
                          f"ATR: {atr_value:.2f} | "
                          f"RSI: {self.rsi[-1]:.1f}")
        
        else:
            # Exit Logic 🛑
            exit_rsi = (self.rsi[-2] >