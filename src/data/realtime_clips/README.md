# 🌙 Moon Dev's Real-Time Clips Agent

**AI-Powered OBS Clip Creator with Model Factory Integration**

Automatically finds the best moments from your live streams while you're recording and names your clips! Works with ALL Moon Dev models through the model_factory.

## ✨ Features

- 🎬 **Autonomous Clipping** - Monitors OBS recordings and creates clips automatically
- 🤖 **AI-Powered** - Uses AI to find best segments, rate quality (1-5), and generate titles
- 🔄 **Model Flexibility** - Works with Groq, Claude, GPT, DeepSeek, Grok, or Ollama
- 🎯 **Smart Filtering** - Only keeps clips scoring 4+ out of 5
- 📱 **Social Media Ready** - Creates both normal and tall (9:16) versions
- 🐦 **Twitter Integration** - Auto-opens compose window with title
- 💰 **Cost Effective** - Uses local Whisper (FREE) for transcription

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Activate your environment
conda activate tflow

# Install required packages
pip install openai-whisper termcolor
```

### 2. Make sure FFmpeg is installed

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### 3. Configure Settings

Edit `/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/config.py`:

```python
# Real-Time Clips Agent Settings 🎬
REALTIME_CLIPS_ENABLED = True
REALTIME_CLIPS_OBS_FOLDER = '/Volumes/Moon 26/OBS'  # Your OBS recording folder
REALTIME_CLIPS_AUTO_INTERVAL = 120  # Check every 2 minutes
REALTIME_CLIPS_LENGTH = 2  # Analyze last 2 minutes
REALTIME_CLIPS_AI_MODEL = 'groq'  # groq, openai, claude, deepseek, xai, ollama
REALTIME_CLIPS_AI_MODEL_NAME = None  # None = use default, or specify model
REALTIME_CLIPS_TWITTER = True  # Auto-open Twitter compose after clip
```

Or edit the agent file directly at the top:
- `OBS_FOLDER` - Your OBS recording directory
- `AI_MODEL_TYPE` - Which AI provider to use (default: 'groq')
- `AUTONOMOUS` - True for auto-mode, False for manual
- `AUTO_CLIP_INTERVAL` - How often to check (seconds)
- `AUTO_CLIP_LENGTH` - How many minutes to analyze

### 4. Run the Agent

```bash
# Navigate to project root
cd /Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading

# Run the agent
python src/agents/realtime_clips_agent.py
```

## 🎮 Usage Modes

### Autonomous Mode (Default)
Automatically checks every N minutes and creates clips when AI rates content 4+:

```python
AUTONOMOUS = True  # At top of agent file
```

The agent will:
1. Check every 2 minutes (configurable)
2. Analyze the last 2 minutes of your stream
3. Ask AI to find the best segment
4. Rate that segment 1-5
5. Only save if score is 4 or 5
6. Generate AI title and create clips
7. Repeat forever until you stop it

### Interactive Mode
Manually request clips of any length:

```python
AUTONOMOUS = False  # At top of agent file
```

Commands:
- `5` or `clip 5` - Create clip from last 5 minutes
- `/help` - Show help
- `/quit` - Exit

## 🤖 AI Model Selection

The agent works with **ALL models** via Moon Dev's Model Factory!

### Recommended Models by Use Case:

| Model | Speed | Cost | Best For |
|-------|-------|------|----------|
| **Groq** | ⚡⚡⚡ | 💰 | Real-time (default!) |
| **Grok (xAI)** | ⚡⚡ | 💰 | Balanced |
| **Claude** | ⚡ | 💰💰 | Quality |
| **GPT-4o** | ⚡⚡ | 💰💰 | Versatile |
| **DeepSeek** | ⚡ | 💰 | Reasoning |
| **Ollama** | ⚡⚡ | FREE | Local/Private |

### Change Model

Edit at top of `realtime_clips_agent.py`:

```python
AI_MODEL_TYPE = 'groq'  # Change to: claude, openai, deepseek, xai, ollama
AI_MODEL_NAME = None    # Or specify: 'gpt-4o', 'claude-3-5-sonnet-latest', etc.
```

## 📂 Output Structure

Clips are saved to `/src/data/realtime_clips/[DATE]/`:

```
src/data/realtime_clips/
├── 10-20-2025/
│   ├── explaining_order_flow_strategies.mov       # Normal clip
│   ├── tall_explaining_order_flow_strategies.mov  # 9:16 vertical
│   ├── building_ai_trading_bots.mov
│   ├── tall_building_ai_trading_bots.mov
│   └── temp_2min_20251020_142030.mov             # Temp (deleted after)
└── 10-21-2025/
    └── ...
```

## 🎯 How AI Rating Works

The agent uses a **7-step workflow**:

1. **Extract** - Last N minutes from OBS recording (ffmpeg)
2. **Transcribe** - Local Whisper creates transcript with timestamps
3. **Find Best Segment** - AI analyzes transcript, trims dead air/filler
4. **Extract Trimmed Text** - Gets just the good content
5. **Rate Quality** - AI scores 1-5 (only keeps 4+)
6. **Generate Title** - AI creates searchable, descriptive title
7. **Create Clips** - Normal + tall versions saved

### Rating Scale (1-5):

- **5** ⭐⭐⭐⭐⭐ - Excellent! Starts strong, valuable insights, engaging
- **4** ⭐⭐⭐⭐ - Good content, interesting information
- **3** ⭐⭐⭐ - Mediocre, some filler mixed with content
- **2** ⭐⭐ - Weak, mostly filler
- **1** ⭐ - Not worth clipping, dead air or off-topic

**Only scores of 4 or 5 are saved!**

## 🎬 Creating Tall Clips

The agent automatically creates 9:16 vertical versions perfect for:
- Twitter/X
- TikTok
- Instagram Reels
- YouTube Shorts

Uses split-screen format:
- Top half: Right side of stream
- Bottom half: Left side of stream
- Both zoomed to fill 1080x960 each = 1080x1920 total

## 🐦 Twitter Integration

When `TWITTER = True`:
1. Clip is created
2. Browser opens Twitter compose
3. Title is pre-filled
4. Drag and drop the clip file
5. Tweet!

## ⚙️ Advanced Configuration

### Prompts

You can customize the AI prompts at the top of the agent file:
- `DECISION_PROMPT` - How AI rates clips (1-5)
- `TRIM_PROMPT` - How AI finds best segments
- `TITLE_PROMPT` - How AI generates titles

### Autonomous Settings

```python
AUTO_CLIP_INTERVAL = 120  # Check every N seconds (120 = 2 min)
AUTO_CLIP_LENGTH = 2      # Analyze last N minutes
```

**Example configurations:**
- **Conservative**: `INTERVAL=300` (5 min), `LENGTH=3` - Fewer, longer clips
- **Aggressive**: `INTERVAL=60` (1 min), `LENGTH=1` - More frequent, shorter clips
- **Balanced** (default): `INTERVAL=120` (2 min), `LENGTH=2` - Good middle ground

## 💡 Tips

1. **Start with Groq** - Fastest and cheapest for real-time use
2. **Check API keys** - Make sure your chosen model's key is in `.env`
3. **Monitor first run** - Let it run a few cycles to see what gets clipped
4. **Adjust interval** - If too many/few clips, tweak `AUTO_CLIP_INTERVAL`
5. **OBS must be recording** - Agent looks for `.mov` files in OBS folder

## 🐛 Troubleshooting

### "No .mov files found"
- Make sure OBS is recording to the correct folder
- Check `OBS_FOLDER` path is correct

### "Model not available"
- Check API key in `.env` file
- Try different model: `AI_MODEL_TYPE = 'groq'`
- Run `python src/models/model_factory.py` to test

### "FFmpeg error"
- Install ffmpeg: `brew install ffmpeg` (macOS)
- Check video file isn't corrupted

### Whisper loading slow
- First load takes time (downloading model)
- Subsequent runs are fast
- Use "base" model for speed (already default)

## 📊 Stats Tracking

In autonomous mode, the agent tracks:
- Total checks
- Clips created
- Clips skipped
- Current score stats

Press `Ctrl+C` to see final stats.

## 🔮 Future Ideas

- [ ] Highlight reels (combine best clips)
- [ ] Custom rating criteria per stream type
- [ ] Multi-stream support
- [ ] Twitch integration
- [ ] Auto-upload to social media
- [ ] Clip length optimization

## 🙏 Credits

Built with love by Moon Dev 🌙

- Uses OpenAI Whisper for transcription
- FFmpeg for video processing
- Moon Dev's Model Factory for AI flexibility
- Inspired by the original clipper from customer-success-ai

---

**Questions? Issues? Ideas?**

Open an issue or ping Moon Dev! 🚀
