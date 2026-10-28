# CrimeVision AI - Multimodal Forensic Analysis Platform
Analyzes image, video and audio evidence (RT-DETR, CLIP, BLIP, Whisper), fuses it into one
chronological timeline, and uses a Groq LLM for crime classification and evidence chat.

## Run
```bash
git clone https://github.com/<your-username>/crimevision-ai.git && cd crimevision-ai
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env     # add GROQ_API_KEY
python app.py            # http://localhost:5000
pytest
```
## REST API
| Method | Endpoint | Purpose |
|---|---|---|
| POST | /api/analyze/{image,video,audio} | multipart `file`, optional `case_id` |
| GET | /api/timeline/<case_id> | fused timeline + detections |
| POST | /api/classify/<case_id> | crime type, scenario, findings |
| POST | /api/chat/<case_id> | `{"question": "..."}` |
| GET | /api/report/<case_id> | PDF report |

## Notes
- COCO-trained RT-DETR detects knives/bats/scissors, not firearms. CLIP zero-shot flags handguns; fine-tune RT-DETR on a weapons dataset for production.
- Decision-support only: all AI output needs qualified human verification.
