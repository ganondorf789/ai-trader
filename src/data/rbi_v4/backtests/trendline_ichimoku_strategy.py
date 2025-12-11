"""
Trendline Breakout + Ichimoku Cloud Strategy
Based on: https://www.tradingview.com/chart/BTCUSD/mBrlcrOV-BTC-USD-4H-CHART-PATTERN/

Strategy Logic:
- Detect downward trendline breakouts
- Confirm with Ichimoku Cloud (price above cloud = bullish)
- Entry: Breakout above trendline + above cloud
- Exit: Price falls below cloud or trailing stop

Timeframe: 4H (as specified in the article)
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


def calculate_trendline(highs, lows, lookback=20, trendline_type='down'):
    """
    Calculate dynamic trendline using swing points

    For downtrend: Connect swing highs
    For uptrend: Connect swing lows
    """
    n = len(highs)
    trendline = np.full(n, np.nan)
    slope_arr = np.full(n, np.nan)

    for i in range(lookback * 2, n):
        if trendline_type == 'down':
            # Find swing highs for downtrend line
            prices = highs[i-lookback*2:i]
            # Find local maxima
            swing_points = []
            for j in range(2, len(prices)-2):
                if prices[j] > prices[j-1] and prices[j] > prices[j-2] and \
                   prices[j] > prices[j+1] and prices[j] > prices[j+2]:
                    swing_points.append((j, prices[j]))

            if len(swing_points) >= 2:
                # Use first and last swing points to draw trendline
                x1, y1 = swing_points[0]
                x2, y2 = swing_points[-1]
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    # Project trendline to current bar
                    current_x = lookback * 2 - 1
                    trendline[i] = y2 + slope * (current_x - x2)
                    slope_arr[i] = slope
        else:
            # Find swing lows for uptrend line
            prices = lows[i-lookback*2:i]
            swing_points = []
            for j in range(2, len(prices)-2):
                if prices[j] < prices[j-1] and prices[j] < prices[j-2] and \
                   prices[j] < prices[j+1] and prices[j] < prices[j+2]:
                    swing_points.append((j, prices[j]))

            if len(swing_points) >= 2:
                x1, y1 = swing_points[0]
                x2, y2 = swing_points[-1]
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    current_x = lookback * 2 - 1
                    trendline[i] = y2 + slope * (current_x - x2)
                    slope_arr[i] = slope

    return trendline, slope_arr


def detect_trendline_breakout(data, lookback=20):
    """
    Detect trendline breakouts

    Returns:
    - breakout_up: 1 when price breaks above downtrend line
    - breakout_down: 1 when price breaks below uptrend line
    - down_trendline: the downward trendline values
    - up_trendline: the upward trendline values
    """
    close = data['Close'].values
    high = data['High'].values
    low = data['Low'].values
    n = len(close)

    breakout_up = np.zeros(n)
    breakout_down = np.zeros(n)

    # Calculate trendlines
    down_trendline, down_slope = calculate_trendline(high, low, lookback, 'down')
    up_trendline, up_slope = calculate_trendline(high, low, lookback, 'up')

    for i in range(lookback * 2 + 1, n):
        # Breakout above downtrend line
        if not np.isnan(down_trendline[i]) and not np.isnan(down_trendline[i-1]):
            if close[i-1] <= down_trendline[i-1] and close[i] > down_trendline[i]:
                if down_slope[i] < 0:  # Confirm it's a downward sloping line
                    breakout_up[i] = 1

        # Breakout below uptrend line
        if not np.isnan(up_trendline[i]) and not np.isnan(up_trendline[i-1]):
            if close[i-1] >= up_trendline[i-1] and close[i] < up_trendline[i]:
                if up_slope[i] > 0:  # Confirm it's an upward sloping line
                    breakout_down[i] = 1

    return (pd.Series(breakout_up, index=data.index),
            pd.Series(breakout_down, index=data.index),
            pd.Series(down_trendline, index=data.index),
            pd.Series(up_trendline, index=data.index))


class TrendlineIchimokuStrategy(Strategy):
    """
    Trendline Breakout + Ichimoku Cloud Confirmation Strategy

    Parameters:
    - trendline_lookback: Lookback period for trendline calculation
    - ichimoku_tenkan: Tenkan-sen period (conversion line)
    - ichimoku_kijun: Kijun-sen period (base line)
    - ichimoku_senkou: Senkou Span B period
    - atr_period: ATR period for stop loss
    - trailing_atr: ATR multiplier for trailing stop
    """

    trendline_lookback = 20
    ichimoku_tenkan = 9
    ichimoku_kijun = 26
    ichimoku_senkou = 52
    atr_period = 14
    trailing_atr = 2.0

    def init(self):
        # Trendline breakout detection
        self.breakout_up, self.breakout_down, self.down_trendline, self.up_trendline = \
            self.I(lambda: detect_trendline_breakout(
                self.data.df, self.trendline_lookback
            ), plot=False)

        # Ichimoku Cloud components
        # Using pandas_ta ichimoku
        df = self.data.df.copy()
        ichimoku_df, _ = ta.ichimoku(
            df['High'], df['Low'], df['Close'],
            tenkan=self.ichimoku_tenkan,
            kijun=self.ichimoku_kijun,
            senkou=self.ichimoku_senkou
        )

        # Extract Ichimoku components
        self.tenkan = self.I(lambda: ichimoku_df[f'ITS_{self.ichimoku_tenkan}'].values, plot=False)
        self.kijun = self.I(lambda: ichimoku_df[f'IKS_{self.ichimoku_kijun}'].values, plot=False)
        self.senkou_a = self.I(lambda: ichimoku_df[f'ISA_{self.ichimoku_tenkan}'].values, plot=False)
        self.senkou_b = self.I(lambda: ichimoku_df[f'ISB_{self.ichimoku_kijun}'].values, plot=False)

        # ATR for stops
        self.atr = self.I(ta.atr,
                          pd.Series(self.data.High),
                          pd.Series(self.data.Low),
                          pd.Series(self.data.Close),
                          self.atr_period)

        # RSI for additional confirmation
        self.rsi = self.I(ta.rsi, pd.Series(self.data.Close), 14)

        # Position tracking
        self.entry_price = None
        self.stop_loss = None
        self.highest_since_entry = None
        self.lowest_since_entry = None

    def next(self):
        if len(self.data) < self.ichimoku_senkou + 30:
            return

        price = self.data.Close[-1]
        atr = self.atr[-1]

        if np.isnan(atr) or atr == 0:
            return

        # Get indicator values
        breakout_up = self.breakout_up[-1]
        breakout_down = self.breakout_down[-1]

        senkou_a = self.senkou_a[-1] if not np.isnan(self.senkou_a[-1]) else price
        senkou_b = self.senkou_b[-1] if not np.isnan(self.senkou_b[-1]) else price
        tenkan = self.tenkan[-1] if not np.isnan(self.tenkan[-1]) else price
        kijun = self.kijun[-1] if not np.isnan(self.kijun[-1]) else price

        rsi = self.rsi[-1] if not np.isnan(self.rsi[-1]) else 50

        # Cloud boundaries
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)

        # Price position relative to cloud
        above_cloud = price > cloud_top
        below_cloud = price < cloud_bottom
        in_cloud = not above_cloud and not below_cloud

        # Bullish cloud (Senkou A > Senkou B)
        bullish_cloud = senkou_a > senkou_b

        # Position management
        if self.position:
            if self.position.is_long:
                if self.highest_since_entry is None:
                    self.highest_since_entry = price
                else:
                    self.highest_since_entry = max(self.highest_since_entry, price)

                # Trailing stop
                trailing_stop = self.highest_since_entry - self.trailing_atr * atr

                # Exit conditions for long
                exit_signal = False

                # 1. Price falls below cloud
                if below_cloud:
                    exit_signal = True

                # 2. Trailing stop hit
                elif price < trailing_stop:
                    exit_signal = True

                # 3. Tenkan crosses below Kijun (bearish signal)
                elif tenkan < kijun and self.tenkan[-2] >= self.kijun[-2]:
                    exit_signal = True

                if exit_signal:
                    self.position.close()
                    self.reset_position_vars()

            elif self.position.is_short:
                if self.lowest_since_entry is None:
                    self.lowest_since_entry = price
                else:
                    self.lowest_since_entry = min(self.lowest_since_entry, price)

                trailing_stop = self.lowest_since_entry + self.trailing_atr * atr

                exit_signal = False

                # 1. Price rises above cloud
                if above_cloud:
                    exit_signal = True

                # 2. Trailing stop hit
                elif price > trailing_stop:
                    exit_signal = True

                # 3. Tenkan crosses above Kijun (bullish signal)
                elif tenkan > kijun and self.tenkan[-2] <= self.kijun[-2]:
                    exit_signal = True

                if exit_signal:
                    self.position.close()
                    self.reset_position_vars()

        else:
            # Entry logic

            # Long entry: Trendline breakout + above cloud + bullish momentum
            if breakout_up == 1:
                bullish_conditions = [
                    above_cloud,  # Price above cloud
                    tenkan > kijun,  # Tenkan above Kijun
                    rsi > 40 and rsi < 75,  # RSI not extreme
                ]

                if sum(bullish_conditions) >= 2:
                    self.entry_price = price
                    self.stop_loss = cloud_bottom - atr
                    self.highest_since_entry = price
                    self.buy()

            # Short entry: Trendline breakdown + below cloud + bearish momentum
            elif breakout_down == 1:
                bearish_conditions = [
                    below_cloud,  # Price below cloud
                    tenkan < kijun,  # Tenkan below Kijun
                    rsi < 60 and rsi > 25,  # RSI not extreme
                ]

                if sum(bearish_conditions) >= 2:
                    self.entry_price = price
                    self.stop_loss = cloud_top + atr
                    self.lowest_since_entry = price
                    self.sell()

            # Additional entry: Strong cloud breakout without trendline
            elif not in_cloud:
                # Recent cloud breakout detection
                prev_above = self.data.Close[-2] > max(self.senkou_a[-2], self.senkou_b[-2])
                prev_below = self.data.Close[-2] < min(self.senkou_a[-2], self.senkou_b[-2])

                # Bullish cloud breakout
                if above_cloud and not prev_above and tenkan > kijun and rsi > 50:
                    self.entry_price = price
                    self.stop_loss = cloud_bottom - atr
                    self.highest_since_entry = price
                    self.buy()

                # Bearish cloud breakdown
                elif below_cloud and not prev_below and tenkan < kijun and rsi < 50:
                    self.entry_price = price
                    self.stop_loss = cloud_top + atr
                    self.lowest_since_entry = price
                    self.sell()

    def reset_position_vars(self):
        self.entry_price = None
        self.stop_loss = None
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
            TrendlineIchimokuStrategy,
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

    results_file = os.path.join(output_dir, f"trendline_ichimoku_results_{timestamp}.json")
    output_data = {
        "strategy": "Trendline Breakout + Ichimoku Cloud Strategy",
        "source": "https://www.tradingview.com/chart/BTCUSD/mBrlcrOV-BTC-USD-4H-CHART-PATTERN/",
        "execution_time": timestamp,
        "timeframe": "4H",
        "parameters": {
            "trendline_lookback": 20,
            "ichimoku_tenkan": 9,
            "ichimoku_kijun": 26,
            "ichimoku_senkou": 52,
            "atr_period": 14,
            "trailing_atr": 2.0
        },
        "summary": summary_stats,
        "per_file_results": all_results
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    csv_file = os.path.join(output_dir, f"trendline_ichimoku_summary_{timestamp}.csv")
    df = pd.DataFrame(all_results)
    df.to_csv(csv_file, index=False)

    print(f"\n  Results saved to: {results_file}")
    print(f"  Summary saved to: {csv_file}")

    return output_data


def main():
    """Main function to run backtests on all OHLCV files"""
    print("="*60)
    print("Trendline Breakout + Ichimoku Cloud Strategy")
    print("Based on TradingView analysis (4H timeframe)")
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
