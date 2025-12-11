"""
Birdeye OHLCV Downloader Service
Upload a token list JSON file and download OHLCV data via Birdeye API
"""

import os
import json
import time
import pandas as pd
import requests
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from pathlib import Path
import zipfile
import io
import threading

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder='templates')
CORS(app)

# Configuration
BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY")
BASE_URL = "https://public-api.birdeye.so/defi"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "ohlcv_downloads"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Valid timeframes for Birdeye API
VALID_TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1H', '2H', '4H', '6H', '8H', '12H', '1D', '3D', '1W', '1M']

# Progress tracking
download_progress = {}


def fetch_ohlcv(address: str, chain: str, timeframe: str, days_back: int) -> pd.DataFrame:
    """
    Fetch OHLCV data from Birdeye API for a single token
    """
    time_to = int(datetime.now().timestamp())
    time_from = int((datetime.now() - timedelta(days=days_back)).timestamp())

    # Birdeye API endpoint
    url = f"{BASE_URL}/ohlcv"
    params = {
        "address": address,
        "type": timeframe,
        "time_from": time_from,
        "time_to": time_to
    }

    headers = {
        "X-API-KEY": BIRDEYE_API_KEY,
        "x-chain": chain if chain else "solana"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)

        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("data", {}).get("items"):
                items = data["data"]["items"]
                df = pd.DataFrame(items)
                # Rename columns to standard OHLCV format
                if 'unixTime' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['unixTime'], unit='s')
                    df = df.rename(columns={
                        'o': 'open',
                        'h': 'high',
                        'l': 'low',
                        'c': 'close',
                        'v': 'volume'
                    })
                return df
            return pd.DataFrame()
        else:
            print(f"API Error {response.status_code}: {response.text}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching OHLCV for {address}: {e}")
        return pd.DataFrame()


def download_all_tokens(task_id: str, tokens: list, timeframe: str, days_back: int):
    """
    Download OHLCV data for all tokens in the list
    """
    global download_progress

    total = len(tokens)
    results = []
    errors = []

    download_progress[task_id] = {
        "status": "running",
        "total": total,
        "completed": 0,
        "current_token": "",
        "errors": [],
        "files": []
    }

    # Create task-specific output directory
    task_dir = OUTPUT_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    for i, token in enumerate(tokens):
        address = token.get("address", "")
        symbol = token.get("symbol", "UNKNOWN")
        chain = token.get("chain", "solana")

        download_progress[task_id]["current_token"] = f"{symbol} ({address[:8]}...)"
        download_progress[task_id]["completed"] = i

        if not address:
            errors.append(f"Missing address for token at index {i}")
            continue

        print(f"[{i+1}/{total}] Downloading {symbol} ({chain})...")

        df = fetch_ohlcv(address, chain, timeframe, days_back)

        if not df.empty:
            filename = f"{symbol}_{chain}_{timeframe}.csv"
            filepath = task_dir / filename
            df.to_csv(filepath, index=False)
            results.append({
                "symbol": symbol,
                "chain": chain,
                "address": address,
                "rows": len(df),
                "file": filename
            })
            download_progress[task_id]["files"].append(filename)
        else:
            errors.append(f"No data for {symbol} ({address})")

        # Rate limiting - avoid hitting API too fast
        time.sleep(0.5)

    download_progress[task_id]["status"] = "completed"
    download_progress[task_id]["completed"] = total
    download_progress[task_id]["current_token"] = ""
    download_progress[task_id]["errors"] = errors
    download_progress[task_id]["results"] = results

    # Create summary file
    summary = {
        "task_id": task_id,
        "timeframe": timeframe,
        "days_back": days_back,
        "total_tokens": total,
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors
    }

    with open(task_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Download complete: {len(results)}/{total} tokens successful")


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html', timeframes=VALID_TIMEFRAMES)


@app.route('/api/upload', methods=['POST'])
def upload_tokens():
    """Handle token list file upload and start download"""
    if not BIRDEYE_API_KEY:
        return jsonify({"error": "BIRDEYE_API_KEY not configured"}), 500

    # Get uploaded file
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Parse JSON file
    try:
        tokens = json.load(file)
        if not isinstance(tokens, list):
            return jsonify({"error": "File must contain a JSON array"}), 400
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400

    # Get parameters
    timeframe = request.form.get('timeframe', '1H')
    if timeframe not in VALID_TIMEFRAMES:
        return jsonify({"error": f"Invalid timeframe. Valid options: {VALID_TIMEFRAMES}"}), 400

    try:
        days_back = int(request.form.get('days', 7))
        if days_back < 1 or days_back > 365:
            return jsonify({"error": "Days must be between 1 and 365"}), 400
    except ValueError:
        return jsonify({"error": "Invalid days value"}), 400

    # Generate task ID
    task_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Start download in background thread
    thread = threading.Thread(
        target=download_all_tokens,
        args=(task_id, tokens, timeframe, days_back)
    )
    thread.start()

    return jsonify({
        "task_id": task_id,
        "total_tokens": len(tokens),
        "timeframe": timeframe,
        "days_back": days_back,
        "message": "Download started"
    })


@app.route('/api/progress/<task_id>')
def get_progress(task_id):
    """Get download progress for a task"""
    if task_id not in download_progress:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(download_progress[task_id])


@app.route('/api/download/<task_id>')
def download_results(task_id):
    """Download all CSV files as a ZIP"""
    task_dir = OUTPUT_DIR / task_id

    if not task_dir.exists():
        return jsonify({"error": "Task results not found"}), 404

    # Create ZIP file in memory
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in task_dir.glob('*'):
            zf.write(file_path, file_path.name)

    memory_file.seek(0)

    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'ohlcv_{task_id}.zip'
    )


@app.route('/api/download/<task_id>/<filename>')
def download_single_file(task_id, filename):
    """Download a single CSV file"""
    file_path = OUTPUT_DIR / task_id / filename

    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404

    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    if not BIRDEYE_API_KEY:
        print("WARNING: BIRDEYE_API_KEY not found in environment!")
        print("Please add BIRDEYE_API_KEY to your .env file")

    print(f"Output directory: {OUTPUT_DIR}")
    print("Starting Birdeye OHLCV Downloader...")
    print("Open http://localhost:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, debug=True)
