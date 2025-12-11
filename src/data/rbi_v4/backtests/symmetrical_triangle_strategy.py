"""
Symmetrical Triangle Breakout Strategy
Based on: https://www.tradingview.com/chart/BTCUSD/zgaZLwbG-BTCUSD-Buyers-Defend-Support-Market-Targets-91-5K-93K-Zone/

Strategy Logic:
- Detect symmetrical triangle pattern (converging trendlines)
- Entry on breakout above descending resistance line
- Support defense confirmation at key levels
- Dual take profit targets (TP1, TP2)
- Stop loss below triangle support

Timeframe: 2H (as specified), adaptable to 4H/1H
Uses backtesting.py with pandas_ta for indicators
"""

import os
import glob
import json
from datetime import datetime
import pandas as pd
import numpy as np
import pandas_ta as ta
from backtesting import Backtest, Strategy


def find_swing_points(data, lookback=5):
    """Find swing highs and swing lows"""
    high = data['High'].values
    low = data['Low'].values
    n = len(high)

    swing_highs = np.full(n, np.nan)
    swing_lows = np.full(n, np.nan)

    for i in range(lookback, n - lookback):
        # Swing high: higher than surrounding bars
        if high[i] == max(high[i-lookback:i+lookback+1]):
            swing_highs[i] = high[i]

        # Swing low: lower than surrounding bars
        if low[i] == min(low[i-lookback:i+lookback+1]):
            swing_lows[i] = low[i]

    return swing_highs, swing_lows


def detect_symmetrical_triangle(data, lookback=30, min_touches=2):
    """
    Detect symmetrical triangle pattern

    Characteristics:
    - Descending resistance line (lower highs)
    - Ascending support line (higher lows)
    - Converging trendlines

    Returns:
    - triangle_detected: 1 when pattern is forming
    - resistance_line: current resistance trendline value
    - support_line: current support trendline value
    - breakout_up: 1 when price breaks above resistance
    - breakout_down: 1 when price breaks below support
    """
    close = data['Close'].values
    high = data['High'].values
    low = data['Low'].values
    n = len(close)

    triangle = np.zeros(n)
    resistance = np.full(n, np.nan)
    support = np.full(n, np.nan)
    breakout_up = np.zeros(n)
    breakout_down = np.zeros(n)

    swing_highs, swing_lows = find_swing_points(data, lookback=5)

    for i in range(lookback * 2, n):
        # Get recent swing points
        recent_highs = []
        recent_lows = []

        for j in range(i - lookback * 2, i):
            if not np.isnan(swing_highs[j]):
                recent_highs.append((j, swing_highs[j]))
            if not np.isnan(swing_lows[j]):
                recent_lows.append((j, swing_lows[j]))

        if len(recent_highs) < min_touches or len(recent_lows) < min_touches:
            continue

        # Calculate resistance line (descending - lower highs)
        if len(recent_highs) >= 2:
            # Use linear regression on swing highs
            x_highs = np.array([p[0] for p in recent_highs])
            y_highs = np.array([p[1] for p in recent_highs])

            if len(x_highs) >= 2:
                slope_h, intercept_h = np.polyfit(x_highs, y_highs, 1)
                resistance[i] = slope_h * i + intercept_h

        # Calculate support line (ascending - higher lows)
        if len(recent_lows) >= 2:
            x_lows = np.array([p[0] for p in recent_lows])
            y_lows = np.array([p[1] for p in recent_lows])

            if len(x_lows) >= 2:
                slope_l, intercept_l = np.polyfit(x_lows, y_lows, 1)
                support[i] = slope_l * i + intercept_l

        # Check for symmetrical triangle (descending resistance + ascending support)
        if not np.isnan(resistance[i]) and not np.isnan(support[i]):
            # Resistance should be descending (negative slope)
            # Support should be ascending (positive slope)
            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                slope_h = (recent_highs[-1][1] - recent_highs[0][1]) / max(1, recent_highs[-1][0] - recent_highs[0][0])
                slope_l = (recent_lows[-1][1] - recent_lows[0][1]) / max(1, recent_lows[-1][0] - recent_lows[0][0])

                # Symmetrical triangle: converging lines
                if slope_h < 0 and slope_l > 0:
                    # Lines should be converging
                    if resistance[i] > support[i]:
                        triangle[i] = 1

                        # Check for breakouts
                        if close[i] > resistance[i] and close[i-1] <= resistance[i-1]:
                            breakout_up[i] = 1
                        elif close[i] < support[i] and close[i-1] >= support[i-1]:
                            breakout_down[i] = 1

    return (pd.Series(triangle, index=data.index),
            pd.Series(resistance, index=data.index),
            pd.Series(support, index=data.index),
            pd.Series(breakout_up, index=data.index),
            pd.Series(breakout_down, index=data.index))


class SymmetricalTriangleStrategy(Strategy):
    """
    Symmetrical Triangle Breakout Strategy

    Parameters:
    - triangle_lookback: Lookback for triangle detection
    - min_touches: Minimum swing points to form trendline
    - atr_period: ATR period for stop loss
    - tp1_atr: ATR multiplier for first target
    - tp2_atr: ATR multiplier for second target
    - stop_atr: ATR multiplier for stop loss
    - partial_exit_pct: Percentage to exit at TP1
    """

    triangle_lookback = 20
    min_touches = 2
    atr_period = 14
    tp1_atr = 1.5  # First target
    tp2_atr = 3.0  # Second target
    stop_atr = 1.5
    partial_exit_pct = 0.5  # Exit 50% at TP1

    def init(self):
        # Triangle detection
        self.triangle, self.resistance, self.support, self.breakout_up, self.breakout_down = \
            self.I(lambda: detect_symmetrical_triangle(
                self.data.df, self.triangle_lookback, self.min_touches
            ), plot=False)

        # ATR for position sizing and stops
        self.atr = self.I(ta.atr,
                          pd.Series(self.data.High),
                          pd.Series(self.data.Low),
                          pd.Series(self.data.Close),
                          self.atr_period)

        # RSI for momentum confirmation
        self.rsi = self.I(ta.rsi, pd.Series(self.data.Close), 14)

        # Volume MA for confirmation
        self.volume_ma = self.I(ta.sma, pd.Series(self.data.Volume), 20)

        # EMA for trend context
        self.ema_20 = self.I(ta.ema, pd.Series(self.data.Close), 20)
        self.ema_50 = self.I(ta.ema, pd.Series(self.data.Close), 50)

        # Position tracking
        self.entry_price = None
        self.stop_loss = None
        self.tp1 = None
        self.tp2 = None
        self.tp1_hit = False
        self.highest_since_entry = None
        self.lowest_since_entry = None

    def next(self):
        if len(self.data) < self.triangle_lookback * 2 + 20:
            return

        price = self.data.Close[-1]
        atr = self.atr[-1]

        if np.isnan(atr) or atr == 0:
            return

        triangle = self.triangle[-1]
        resistance = self.resistance[-1]
        support = self.support[-1]
        breakout_up = self.breakout_up[-1]
        breakout_down = self.breakout_down[-1]

        rsi = self.rsi[-1] if not np.isnan(self.rsi[-1]) else 50
        volume = self.data.Volume[-1]
        volume_ma = self.volume_ma[-1] if not np.isnan(self.volume_ma[-1]) else volume
        ema_20 = self.ema_20[-1] if not np.isnan(self.ema_20[-1]) else price
        ema_50 = self.ema_50[-1] if not np.isnan(self.ema_50[-1]) else price

        # Position management
        if self.position:
            if self.position.is_long:
                if self.highest_since_entry is None:
                    self.highest_since_entry = price
                else:
                    self.highest_since_entry = max(self.highest_since_entry, price)

                # Check TP1
                if not self.tp1_hit and self.tp1 and price >= self.tp1:
                    self.tp1_hit = True
                    # Move stop to breakeven
                    self.stop_loss = self.entry_price

                # Check TP2 or trailing stop
                if self.tp2 and price >= self.tp2:
                    self.position.close()
                    self.reset_position_vars()
                elif self.stop_loss and price <= self.stop_loss:
                    self.position.close()
                    self.reset_position_vars()
                # Trailing stop after TP1
                elif self.tp1_hit:
                    trailing_stop = self.highest_since_entry - 1.5 * atr
                    if price < trailing_stop:
                        self.position.close()
                        self.reset_position_vars()

            elif self.position.is_short:
                if self.lowest_since_entry is None:
                    self.lowest_since_entry = price
                else:
                    self.lowest_since_entry = min(self.lowest_since_entry, price)

                # Check TP1
                if not self.tp1_hit and self.tp1 and price <= self.tp1:
                    self.tp1_hit = True
                    self.stop_loss = self.entry_price

                # Check TP2 or trailing stop
                if self.tp2 and price <= self.tp2:
                    self.position.close()
                    self.reset_position_vars()
                elif self.stop_loss and price >= self.stop_loss:
                    self.position.close()
                    self.reset_position_vars()
                elif self.tp1_hit:
                    trailing_stop = self.lowest_since_entry + 1.5 * atr
                    if price > trailing_stop:
                        self.position.close()
                        self.reset_position_vars()

        else:
            # Entry logic

            # Long entry: Breakout above triangle resistance
            if breakout_up == 1 or (triangle == 1 and not np.isnan(resistance) and price > resistance):
                # Confirmation filters
                volume_confirm = volume > volume_ma * 1.1  # Volume surge
                momentum_confirm = rsi > 45 and rsi < 75
                trend_confirm = price > ema_20

                if sum([volume_confirm, momentum_confirm, trend_confirm]) >= 2:
                    self.entry_price = price
                    self.stop_loss = support - self.stop_atr * atr if not np.isnan(support) else price - self.stop_atr * atr
                    self.tp1 = price + self.tp1_atr * atr
                    self.tp2 = price + self.tp2_atr * atr
                    self.tp1_hit = False
                    self.highest_since_entry = price
                    self.buy()

            # Short entry: Breakdown below triangle support
            elif breakout_down == 1 or (triangle == 1 and not np.isnan(support) and price < support):
                volume_confirm = volume > volume_ma * 1.1
                momentum_confirm = rsi < 55 and rsi > 25
                trend_confirm = price < ema_20

                if sum([volume_confirm, momentum_confirm, trend_confirm]) >= 2:
                    self.entry_price = price
                    self.stop_loss = resistance + self.stop_atr * atr if not np.isnan(resistance) else price + self.stop_atr * atr
                    self.tp1 = price - self.tp1_atr * atr
                    self.tp2 = price - self.tp2_atr * atr
                    self.tp1_hit = False
                    self.lowest_since_entry = price
                    self.sell()

    def reset_position_vars(self):
        self.entry_price = None
        self.stop_loss = None
        self.tp1 = None
        self.tp2 = None
        self.tp1_hit = False
        self.highest_since_entry = None
        self.lowest_since_entry = None


def load_and_prepare_data(file_path):
    """Load CSV and prepare for backtesting.py format"""
    df = pd.read_csv(file_path)

    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'timestamp': 'Datetime'
    })

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df[(df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0)]
    df = df.sort_index()

    return df


def run_backtest(file_path, cash=100000, commission=0.001):
    """Run backtest on a single file"""
    print(f"\n{'='*60}")
    print(f"Backtesting: {os.path.basename(file_path)}")
    print(f"{'='*60}")

    try:
        df = load_and_prepare_data(file_path)

        if len(df) < 100:
            print(f"  Skipping: Not enough data ({len(df)} rows)")
            return None

        print(f"  Data range: {df.index[0]} to {df.index[-1]}")
        print(f"  Total bars: {len(df)}")
        print(f"  Price range: {df['Close'].min():.2f} - {df['Close'].max():.2f}")

        bt = Backtest(
            df,
            SymmetricalTriangleStrategy,
            cash=cash,
            commission=commission,
            exclusive_orders=True
        )

        stats = bt.run()
        return stats

    except Exception as e:
        print(f"  Error: {str(e)}")
        return None


def format_stats(stats):
    """Format backtest statistics for display"""
    if stats is None:
        return "No results"

    output = []
    output.append(f"  Return [%]:              {stats['Return [%]']:.2f}%")
    output.append(f"  Buy & Hold Return [%]:   {stats['Buy & Hold Return [%]']:.2f}%")
    output.append(f"  Max Drawdown [%]:        {stats['Max. Drawdown [%]']:.2f}%")
    output.append(f"  # Trades:                {stats['# Trades']}")

    if stats['# Trades'] > 0:
        output.append(f"  Win Rate [%]:            {stats['Win Rate [%]']:.2f}%")
        pf = stats['Profit Factor']
        output.append(f"  Profit Factor:           {pf:.2f}" if not np.isnan(pf) else "  Profit Factor:           N/A")
        sr = stats['Sharpe Ratio']
        output.append(f"  Sharpe Ratio:            {sr:.2f}" if not np.isnan(sr) else "  Sharpe Ratio:            N/A")
        output.append(f"  Avg Trade [%]:           {stats['Avg. Trade [%]']:.2f}%")
        output.append(f"  Best Trade [%]:          {stats['Best Trade [%]']:.2f}%")
        output.append(f"  Worst Trade [%]:         {stats['Worst Trade [%]']:.2f}%")

    output.append(f"  Final Equity:            ${stats['Equity Final [$]']:,.2f}")

    return "\n".join(output)


def save_results(all_results, summary_stats, output_dir):
    """Save backtest results to execution_results directory"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_file = os.path.join(output_dir, f"symmetrical_triangle_results_{timestamp}.json")
    output_data = {
        "strategy": "Symmetrical Triangle Breakout Strategy",
        "source": "https://www.tradingview.com/chart/BTCUSD/zgaZLwbG-BTCUSD-Buyers-Defend-Support-Market-Targets-91-5K-93K-Zone/",
        "execution_time": timestamp,
        "timeframe": "2H/4H",
        "parameters": {
            "triangle_lookback": 20,
            "min_touches": 2,
            "atr_period": 14,
            "tp1_atr": 1.5,
            "tp2_atr": 3.0,
            "stop_atr": 1.5
        },
        "summary": summary_stats,
        "per_file_results": all_results
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    csv_file = os.path.join(output_dir, f"symmetrical_triangle_summary_{timestamp}.csv")
    df = pd.DataFrame(all_results)
    df.to_csv(csv_file, index=False)

    print(f"\n  Results saved to: {results_file}")
    print(f"  Summary saved to: {csv_file}")

    return output_data


def main():
    """Main function to run backtests on all OHLCV files"""
    print("="*60)
    print("Symmetrical Triangle Breakout Strategy")
    print("Based on TradingView analysis")
    print("="*60)

    base_dir = os.path.dirname(__file__)
    ohlcv_dir = os.path.join(base_dir, "..", "ohlcv")
    output_dir = os.path.join(base_dir, "..", "execution_results")
    ohlcv_files = glob.glob(os.path.join(ohlcv_dir, "*.csv"))

    if not ohlcv_files:
        print(f"No CSV files found in {ohlcv_dir}")
        return None, None

    print(f"\nFound {len(ohlcv_files)} data file(s)")

    all_results = []

    for file_path in ohlcv_files:
        bt_stats = run_backtest(file_path)

        if bt_stats is not None:
            print("\n  Results:")
            print(format_stats(bt_stats))

            all_results.append({
                'file': os.path.basename(file_path),
                'return': float(bt_stats['Return [%]']),
                'buy_hold_return': float(bt_stats['Buy & Hold Return [%]']),
                'trades': int(bt_stats['# Trades']),
                'win_rate': float(bt_stats['Win Rate [%]']) if bt_stats['# Trades'] > 0 else 0,
                'max_dd': float(bt_stats['Max. Drawdown [%]']),
                'sharpe': float(bt_stats['Sharpe Ratio']) if not np.isnan(bt_stats['Sharpe Ratio']) else 0,
                'profit_factor': float(bt_stats['Profit Factor']) if not np.isnan(bt_stats['Profit Factor']) else 0,
                'avg_trade': float(bt_stats['Avg. Trade [%]']) if bt_stats['# Trades'] > 0 else 0,
                'final_equity': float(bt_stats['Equity Final [$]'])
            })

    if all_results:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        df_results = pd.DataFrame(all_results)

        summary_stats = {
            'total_files': len(df_results),
            'avg_return': float(df_results['return'].mean()),
            'total_trades': int(df_results['trades'].sum()),
            'avg_win_rate': float(df_results['win_rate'].mean()),
            'avg_max_dd': float(df_results['max_dd'].mean()),
            'avg_sharpe': float(df_results['sharpe'].mean()),
            'avg_profit_factor': float(df_results['profit_factor'].mean())
        }

        print(f"\nTotal files tested: {summary_stats['total_files']}")
        print(f"Average Return: {summary_stats['avg_return']:.2f}%")
        print(f"Total Trades: {summary_stats['total_trades']}")
        print(f"Average Win Rate: {summary_stats['avg_win_rate']:.2f}%")
        print(f"Average Max Drawdown: {summary_stats['avg_max_dd']:.2f}%")
        print(f"Average Sharpe Ratio: {summary_stats['avg_sharpe']:.2f}")

        print("\n" + "-"*60)
        print("Per-file Summary (Top 10 by return):")
        print("-"*60)
        df_sorted = df_results.sort_values('return', ascending=False).head(10)
        for _, row in df_sorted.iterrows():
            print(f"  {row['file']:<25} Return: {row['return']:>8.2f}%  Trades: {row['trades']:>3}  Win: {row['win_rate']:>5.1f}%")

        output_data = save_results(all_results, summary_stats, output_dir)

        return all_results, output_data

    return None, None


if __name__ == "__main__":
    results, output_data = main()
