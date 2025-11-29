Here's the complete backtesting implementation for the MomentumVolatility strategy:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import resample_apply

# 🌙 MOON DEV DATA PREPARATION 🌙
def prepare_data(path):
    # Load and clean data
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    
    # Column mapping and validation
    required_columns = {'open': 'Open', 'high': 'High', 'low': 'Low', 
                       'close': 'Close', 'volume': 'Volume', 'vix': 'VIX'}
    data.rename(columns=required_columns, inplace=True)
    
    if not all(col in data.columns for col in ['VIX']):
        raise ValueError("🚨 Missing required VIX column in dataset")
    
    # Convert and index datetime
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

# 🚀 MOON DEV STRATEGY IMPLEMENTATION 🚀
class MomentumVolatility(Strategy):
    rsi_period = 14
    vix_threshold = 15
    sma_period = 200
    risk_pct = 0.02
    stop_loss_pct = 0.03
    corr_window = 20

    def init(self):
        # 🌙 INDICATOR CALCULATIONS 🌙
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period)
        self.sma200 = self.I(talib.SMA, self.data.Close, self.sma_period)
        self.vix = self.I(talib.MA, self.data.VIX, timeperiod=5)  # Smoothed VIX
        self.correlation = self.I(talib.CORREL, self.vix, self.sma200, self.corr_window)

    def next(self):
        # 🛑 EARLY EXIT IF INDICATORS NOT READY 🛑
        if len(self.data) < max(self.sma_period, self.corr_window):
            return

        # 🌙 CURRENT VALUES 🌙
        price = self.data.Close[-1]
        rsi = self.rsi[-1]
        vix = self.vix[-1]
        sma = self.sma200[-1]
        corr = self.correlation[-1]

        # ✨ DEBUG PRINTING ✨
        print((f"🌙 RSI: {rsi:.1f} | VIX: {vix:.1f} | "
               f"Price/SMA: {price:.1f}/{sma:.1f} | Corr: {corr:.2f}"))

        # 🚀 ENTRY LOGIC 🚀
        if not self.position:
            if all([
                rsi > 70,
                vix < self.vix_threshold,
                price > sma
            ]):
                # 🌙 RISK MANAGEMENT CALCULATIONS 🌙
                risk_amount = self.equity * self.risk_pct
                stop_price = price * (1 - self.stop_loss_pct)
                risk_per_share = price - stop_price
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.buy(size=position_size, sl=stop_price)
                        print((f"🚀 MOON ROCKET LAUNCH! "
                               f"Buy {position_size} @ {price:.1f} "
                               f"SL: {stop_price:.1f}"))

        # 🌑 EXIT LOGIC 🌑
        else:
            if any([
                corr < 0,
                rsi < 50,
                self.data.Close[-1] <= self.position.sl
            ]):
                self.position.close()
                print(f"🌑 MOON LANDING! Exit @ {self.data.Close[-1]:.1f}")

# 🌙 BACKTEST EXECUTION 🌙
if __name__ == '__main__':
    # Prepare lunar data
    data = prepare_data('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
    
    # Launch moon mission
    bt = Backtest(data, MomentumVolatility, cash=1_000_000, commission=.002)
    stats = bt.run()
    
    # Print mission results
    print("\n🌕 MO