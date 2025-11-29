# Real-Time Clips Agent

Auto-clips best moments from OBS recordings while streaming.

## What It Does
- Monitors OBS recording folder
- Extracts last N minutes
- AI rates content quality (1-5)
- Creates clips for scores 4+
- Generates titles automatically

## Usage
```bash
python src/agents/realtime_clips_agent.py
```

## Configuration
Top of file:
```python
AI_MODEL_TYPE = 'groq'  # Fast and cheap
AUTONOMOUS = True       # Auto-mode
AUTO_CLIP_INTERVAL = 120  # Check every 2 min
AUTO_CLIP_LENGTH = 2    # Analyze last 2 min
OBS_FOLDER = '/path/to/obs'
```

## Output Formats
- Normal clip (16:9)
- Tall clip (9:16) for socials
- Auto-deletes temp files

## Requirements
- FFmpeg installed
- Whisper for transcription (auto-downloads)

## Output
`src/data/realtime_clips/[date]/[title].mov`