# StoryLlama - Project Context & Implementation Guide

## Project Overview
Build a highly personalized AI-powered caption generator that combines ML-based image understanding with LLM-powered content generation. The system analyzes images, extracts keywords, and generates customized captions, stories, and poems based on user preferences.

---

## Core Features & Requirements

### 1. Caption Generation (Primary Feature)
- **Input**: Single or multiple images (up to 10)
- **Output**: 3 caption variations for user selection
- **Process Flow**:
  1. ML model analyzes image → extracts keywords, objects, actions
  2. Keywords + user preferences → LLM API (Gemini)
  3. LLM generates 3 caption options
  4. Display results with copy/save functionality

### 2. Style & Tone Controls
**Preset Styles:**
- Influencer (casual, engaging, hashtag-friendly)
- Marketing (persuasive, brand-focused, CTA-driven)
- Personal (authentic, storytelling)
- Professional (formal, informative)

**Tone Options:**
- Fun (playful, energetic)
- Formal (professional, serious)
- Emotional (heartfelt, inspiring)
- Funny (humorous, witty)

**Temperature Control:** Slider (0.1 - 1.0)
- Low (0.1-0.4): Predictable, consistent
- Medium (0.5-0.7): Balanced creativity
- High (0.8-1.0): Creative, unexpected

### 3. Advanced Options Panel
- **Length Control**: Word count selector (10-100 words)
- **Context Input**: Text field for additional image context
- **Target Audience**: Demographic/interest specification
- **Custom Keywords**: User can add/remove keywords

### 4. Content Type Selection
- **Caption** (default, 1-3 sentences)
- **Story** (narrative format, 100-300 words)
- **Poem** (creative, structured verses)

### 5. Multi-Image Support
- Upload up to 10 images
- ML model analyzes all images collectively
- LLM creates cohesive narrative across images
- Use case: Photo series, before/after, journey documentation

### 6. "About the Image" Feature
- Descriptive analysis of image contents
- Objects, colors, composition, mood detection
- Educational/accessibility use case

### 7. User Profile System
**Profile Data:**
- User preferences (default style, tone, length)
- Brand voice guidelines (for businesses/influencers)
- Custom vocabulary/hashtags
- Caption history & favorites
- Target audience presets

**Benefits:**
- Faster caption generation (pre-filled preferences)
- Consistent brand voice
- Learning from user's selection patterns

### 8. Story Video Creator
- Select 2-10 images from uploaded set
- Add generated story/captions as overlays
- Transition effects between images
- Background music options (optional)
- Export as MP4 video

---

## Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND (React/Next.js)              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Upload  │  │  Style   │  │ Advanced │  │ Profile  │   │
│  │  Images  │  │ Selector │  │ Options  │  │ Manager  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓ API Calls
┌─────────────────────────────────────────────────────────────┐
│                    BACKEND API (FastAPI/Node.js)             │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │  Image Upload    │  │  User Auth &     │                │
│  │  Handler         │  │  Profile Service │                │
│  └──────────────────┘  └──────────────────┘                │
└─────────────────────────────────────────────────────────────┘
              ↓                              ↓
┌──────────────────────────┐    ┌──────────────────────────┐
│   ML MODEL SERVICE       │    │   DATABASE               │
│   (Python/PyTorch)       │    │   (PostgreSQL)           │
│                          │    │                          │
│  ┌────────────────────┐ │    │  - User profiles         │
│  │ Image Preprocessing│ │    │  - Caption history       │
│  └────────────────────┘ │    │  - Preferences           │
│  ┌────────────────────┐ │    │  - Saved captions        │
│  │ Object Detection   │ │    └──────────────────────────┘
│  │ (YOLO/Faster-RCNN) │ │
│  └────────────────────┘ │
│  ┌────────────────────┐ │
│  │ Feature Extraction │ │
│  │ (CLIP/BLIP-2)      │ │
│  └────────────────────┘ │
│  ┌────────────────────┐ │
│  │ Keyword Generation │ │
│  └────────────────────┘ │
└──────────────────────────┘
              ↓
┌──────────────────────────┐
│   LLM ORCHESTRATOR       │
│                          │
│  ┌────────────────────┐ │
│  │ Prompt Builder     │ │
│  │ - Style injection  │ │
│  │ - Tone adaptation  │ │
│  │ - Context merging  │ │
│  └────────────────────┘ │
│            ↓             │
│  ┌────────────────────┐ │
│  │ Gemini API Client  │ │
│  │ (or GPT/Claude)    │ │
│  └────────────────────┘ │
│            ↓             │
│  ┌────────────────────┐ │
│  │ Response Parser    │ │
│  │ Generate 3 options │ │
│  └────────────────────┘ │
└──────────────────────────┘
```

### Data Flow

```
1. USER UPLOADS IMAGE(S)
   ↓
2. FRONTEND → BACKEND
   - FormData with images
   - User preferences (style, tone, temperature)
   - Advanced options (if provided)
   ↓
3. BACKEND → ML MODEL SERVICE
   - Image preprocessing
   - Run inference
   - Extract: objects, actions, colors, mood, composition
   ↓
4. ML MODEL RETURNS KEYWORDS
   Example: {
     "objects": ["sunset", "beach", "person", "surfboard"],
     "actions": ["surfing", "standing"],
     "colors": ["orange", "blue", "golden"],
     "mood": "peaceful, adventurous",
     "composition": "rule of thirds, silhouette"
   }
   ↓
5. BACKEND → LLM ORCHESTRATOR
   - Build dynamic prompt
   - Inject user preferences
   - Add context if provided
   ↓
6. LLM ORCHESTRATOR → GEMINI API
   Prompt Example:
   """
   You are a {style} content creator.
   Tone: {tone}
   Temperature: {temp}
   
   Image Analysis:
   - Objects detected: {objects}
   - Actions: {actions}
   - Mood: {mood}
   - Colors: {colors}
   
   User Context: {user_context}
   Target Audience: {audience}
   
   Generate 3 {content_type} options ({length} words each).
   Each should be unique and {tone}.
   
   Format as JSON:
   {
     "option1": "...",
     "option2": "...",
     "option3": "..."
   }
   """
   ↓
7. GEMINI RETURNS 3 CAPTIONS
   ↓
8. BACKEND → FRONTEND
   - Display 3 options
   - User selects favorite
   - Option to save to profile
```

---

## Technology Stack

### Frontend
```
Framework: Next.js 14 (App Router)
Language: TypeScript
Styling: TailwindCSS + Shadcn/ui
State: Zustand or React Context
Image Upload: react-dropzone
Forms: react-hook-form + zod validation
```

### Backend
```
API Framework: FastAPI (Python) OR Express.js (Node.js)
Language: Python 3.10+ OR Node.js 18+
Authentication: JWT tokens, NextAuth
Database ORM: SQLAlchemy (Python) OR Prisma (Node.js)
File Storage: AWS S3, Cloudinary, or local storage
```

### ML Model
```
Framework: PyTorch OR TensorFlow
Pre-trained Models:
  - Image Captioning: BLIP-2, CLIP
  - Object Detection: YOLOv8, Faster R-CNN
  - Feature Extraction: ResNet-50, EfficientNet
Libraries:
  - Hugging Face Transformers
  - OpenCV (image preprocessing)
  - Pillow (image manipulation)
Training Dataset: MS COCO, Flickr30k, Conceptual Captions
```

### LLM Integration
```
Primary: Google Gemini API (gemini-1.5-pro or gemini-1.5-flash)
Backup: OpenAI GPT-4, Anthropic Claude
SDK: google-generativeai (Python) or @google/generative-ai (Node.js)
Rate Limiting: Token bucket algorithm
Caching: Redis for repeated queries
```

### Database
```
Primary DB: PostgreSQL 15+
Schema:
  - users (id, email, name, preferences)
  - profiles (user_id, style_preferences, brand_voice)
  - captions (id, user_id, image_url, keywords, generated_text)
  - saved_captions (user_id, caption_id, favorited_at)
Cache: Redis for session/temporary data
```

### Video Generation
```
Library: FFmpeg (Python bindings: ffmpeg-python)
Process:
  1. Stitch images with transitions
  2. Add text overlays (captions/story)
  3. Add background music (optional)
  4. Export as MP4
Alternative: Remotion (React-based video creation)
```

### Deployment
```
Frontend: Vercel, Netlify
Backend/API: Google Cloud Run, AWS Lambda, Railway
ML Model: Hugging Face Spaces, AWS SageMaker, Google Vertex AI
Database: Supabase, Railway, AWS RDS
CDN: Cloudflare (for images/videos)
```

---

## Project Structure

```
ai-caption-generator/
├── frontend/                      # Next.js application
│   ├── src/
│   │   ├── app/                   # App router pages
│   │   │   ├── page.tsx           # Home/upload page
│   │   │   ├── generate/          # Caption generation page
│   │   │   ├── profile/           # User profile
│   │   │   └── api/               # API routes (if using Next.js API)
│   │   ├── components/
│   │   │   ├── ui/                # Shadcn components
│   │   │   ├── ImageUpload.tsx
│   │   │   ├── StyleSelector.tsx
│   │   │   ├── ToneControl.tsx
│   │   │   ├── AdvancedOptions.tsx
│   │   │   ├── CaptionDisplay.tsx
│   │   │   └── ProfileManager.tsx
│   │   ├── lib/
│   │   │   ├── api-client.ts      # Backend API calls
│   │   │   ├── utils.ts
│   │   │   └── types.ts
│   │   └── stores/
│   │       └── caption-store.ts   # Zustand store
│   ├── public/
│   ├── package.json
│   └── next.config.js
│
├── backend/                       # FastAPI or Express.js
│   ├── app/
│   │   ├── main.py                # Entry point
│   │   ├── config.py              # Environment config
│   │   ├── routes/
│   │   │   ├── images.py          # Image upload endpoints
│   │   │   ├── captions.py        # Caption generation endpoints
│   │   │   ├── users.py           # User/profile endpoints
│   │   │   └── videos.py          # Video creation endpoints
│   │   ├── services/
│   │   │   ├── ml_service.py      # ML model integration
│   │   │   ├── llm_service.py     # LLM orchestration
│   │   │   ├── prompt_builder.py  # Dynamic prompt construction
│   │   │   └── video_service.py   # FFmpeg video creation
│   │   ├── models/                # Database models
│   │   │   ├── user.py
│   │   │   ├── caption.py
│   │   │   └── profile.py
│   │   └── utils/
│   │       ├── image_processor.py
│   │       └── validators.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── ml-model/                      # ML model service
│   ├── models/
│   │   ├── image_captioning.py    # BLIP-2/CLIP model
│   │   ├── object_detection.py    # YOLO model
│   │   └── feature_extractor.py   # ResNet/EfficientNet
│   ├── preprocessing/
│   │   └── image_preprocessor.py
│   ├── inference/
│   │   └── inference_engine.py
│   ├── training/                  # (Optional) Fine-tuning scripts
│   │   ├── train.py
│   │   └── dataset_loader.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── shared/                        # Shared types/constants
│   └── types.ts
│
├── docker-compose.yml             # Local development setup
├── .env.example                   # Environment variables template
└── README.md
```

---

## Implementation Steps (Phase-by-Phase)

### Phase 1: MVP - Basic Caption Generation (Week 1-2)

**Step 1.1: Setup Project Structure**
```bash
# Initialize monorepo
mkdir ai-caption-generator
cd ai-caption-generator

# Frontend
npx create-next-app@latest frontend --typescript --tailwind --app
cd frontend && npm install zustand react-dropzone react-hook-form zod

# Backend
mkdir backend && cd backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install fastapi uvicorn python-multipart pillow google-generativeai

# ML Model
mkdir ml-model && cd ml-model
pip install torch torchvision transformers opencv-python
```

**Step 1.2: Build Image Upload UI**
- Create ImageUpload component with drag-and-drop
- Preview uploaded images
- Send to backend on submit

**Step 1.3: Setup ML Model Service**
- Load pre-trained BLIP-2 or CLIP model
- Create inference endpoint
- Extract keywords from image
- Return structured JSON

**Step 1.4: Integrate Gemini API**
- Setup Gemini API client
- Create basic prompt template
- Generate 1 caption (test)
- Display in UI

**Step 1.5: Generate 3 Caption Variations**
- Modify prompt to request 3 options
- Parse JSON response
- Display options with copy buttons

**Deliverable:** Working single-image caption generator with 3 options

---

### Phase 2: Style, Tone & Advanced Options (Week 3-4)

**Step 2.1: Style Selector UI**
- Radio buttons or dropdown for styles
- Pass selected style to backend

**Step 2.2: Tone Controls**
- Tone selector (4 options)
- Temperature slider (0.1 - 1.0)

**Step 2.3: Dynamic Prompt Builder**
- Create prompt templates for each style
- Inject tone and temperature
- Test variations

**Step 2.4: Advanced Options Panel**
- Length control (word count)
- Context input field
- Target audience field
- Send to prompt builder

**Step 2.5: Prompt Engineering**
```python
def build_prompt(keywords, style, tone, temperature, length, context, audience):
    style_prompts = {
        "influencer": "You are a social media influencer creating engaging content...",
        "marketing": "You are a marketing expert crafting persuasive copy...",
    }
    
    tone_modifiers = {
        "fun": "Make it playful and energetic.",
        "formal": "Keep it professional and polished.",
    }
    
    prompt = f"""
    {style_prompts[style]}
    {tone_modifiers[tone]}
    
    Image contains: {keywords}
    Additional context: {context}
    Target audience: {audience}
    
    Generate {length} word captions (3 variations) in JSON format.
    """
    return prompt
```

**Deliverable:** Fully customizable caption generation with multiple styles/tones

---

### Phase 3: Multi-Image & Content Types (Week 5-6)

**Step 3.1: Multi-Image Upload**
- Allow up to 10 images
- Process all images through ML model
- Combine keywords from all images

**Step 3.2: Story Generation**
- Add content type selector (Caption/Story/Poem)
- Create story-specific prompts
- Handle longer outputs (100-300 words)

**Step 3.3: Poem Generation**
- Poem structure prompts
- Rhyme scheme options
- Display with proper formatting

**Step 3.4: "About the Image" Feature**
- Descriptive analysis endpoint
- Display in modal or separate section

**Deliverable:** Multi-image support + story/poem generation

---

### Phase 4: User Profiles & Personalization (Week 7-8)

**Step 4.1: Database Setup**
```sql
-- PostgreSQL schema
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE profiles (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    default_style VARCHAR(50),
    default_tone VARCHAR(50),
    default_temperature FLOAT,
    brand_voice TEXT,
    target_audience TEXT
);

CREATE TABLE captions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    image_urls TEXT[],
    keywords JSONB,
    generated_text TEXT,
    style VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE saved_captions (
    user_id INTEGER REFERENCES users(id),
    caption_id INTEGER REFERENCES captions(id),
    favorited_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (user_id, caption_id)
);
```

**Step 4.2: Authentication**
- NextAuth setup
- Login/signup pages
- JWT tokens

**Step 4.3: Profile Manager UI**
- Edit preferences
- Save brand voice
- View caption history

**Step 4.4: Personalization Engine**
- Load user preferences on generation
- Learn from user's selections
- Suggest improvements

**Deliverable:** Full user profile system with history

---

### Phase 5: Video Creator (Week 9-10)

**Step 5.1: Video Creator UI**
- Image selection from uploaded set
- Drag to reorder
- Transition type selector

**Step 5.2: FFmpeg Integration**
```python
import ffmpeg

def create_story_video(images, captions, transitions):
    # Create video from images
    input_streams = [ffmpeg.input(img) for img in images]
    
    # Add transitions
    video = ffmpeg.concat(*input_streams, v=1, a=0)
    
    # Add text overlays
    video = video.drawtext(
        text=captions[0],
        fontsize=24,
        x='(w-text_w)/2',
        y='h-th-10'
    )
    
    # Export
    output = ffmpeg.output(video, 'output.mp4')
    ffmpeg.run(output)
```

**Step 5.3: Export & Download**
- Generate video on server
- Return download link
- Upload to CDN (optional)

**Deliverable:** Story video creator with captions

---

## API Endpoints Specification

### Image Upload & Processing
```
POST /api/upload
Body: FormData { images: File[] }
Response: { image_urls: string[] }

POST /api/analyze
Body: { image_urls: string[] }
Response: { keywords: KeywordObject }
```

### Caption Generation
```
POST /api/generate-caption
Body: {
  image_urls: string[],
  keywords: KeywordObject,
  style: string,
  tone: string,
  temperature: number,
  length?: number,
  context?: string,
  audience?: string
}
Response: {
  options: [string, string, string]
}

POST /api/generate-story
Body: { ... } (same as caption)
Response: { story: string }

POST /api/generate-poem
Body: { ... }
Response: { poem: string }

GET /api/about-image?image_url=...
Response: { description: string }
```

### User & Profile
```
POST /api/auth/register
POST /api/auth/login
GET /api/profile
PUT /api/profile
GET /api/captions/history
POST /api/captions/save
```

### Video Creation
```
POST /api/video/create
Body: {
  image_urls: string[],
  captions: string[],
  transitions: string[]
}
Response: { video_url: string }
```

---

## Environment Variables

```env
# Frontend (.env.local)
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_GEMINI_API_KEY=your_key_here  # (if client-side calls)

# Backend (.env)
DATABASE_URL=postgresql://user:pass@localhost:5432/captiondb
REDIS_URL=redis://localhost:6379
GEMINI_API_KEY=your_gemini_key
AWS_S3_BUCKET=your-bucket-name
AWS_ACCESS_KEY=your-access-key
AWS_SECRET_KEY=your-secret-key
JWT_SECRET=your-jwt-secret

# ML Model Service (.env)
MODEL_CACHE_DIR=/models
TORCH_DEVICE=cuda  # or cpu
```

---

## Prompt Templates (Examples)

### Influencer Style - Fun Tone
```
You are a popular social media influencer with 500K followers.
Your content is always energetic, relatable, and engaging.

Image Analysis:
- Objects: {objects}
- Actions: {actions}
- Mood: {mood}

Create 3 Instagram captions (30 words each) that:
1. Start with an attention-grabbing hook
2. Include relevant emojis
3. End with a call-to-action or question
4. Are playful and fun

Format as JSON: {"option1": "...", "option2": "...", "option3": "..."}
```

### Marketing Style - Formal Tone
```
You are a professional copywriter for a premium brand.
Your tone is sophisticated, persuasive, and benefit-focused.

Image Analysis:
- Objects: {objects}
- Target Audience: {audience}
- Brand Context: {context}

Create 3 marketing captions (50 words each) that:
1. Highlight key benefits
2. Use power words
3. Include a clear CTA
4. Maintain professional tone

Format as JSON: {"option1": "...", "option2": "...", "option3": "..."}
```

---

## Testing Strategy

### Unit Tests
- ML model inference accuracy
- Prompt builder output validation
- API endpoint responses

### Integration Tests
- End-to-end caption generation flow
- Multi-image processing
- Video creation pipeline

### User Testing
- A/B test different prompt styles
- Measure caption selection rates
- Gather feedback on generated content quality

---

## Performance Optimization

1. **ML Model Optimization**
   - Use quantized models (INT8)
   - Batch processing for multiple images
   - GPU acceleration

2. **Caching Strategy**
   - Cache ML model outputs (keywords)
   - Redis for repeated LLM queries
   - CDN for images/videos

3. **Rate Limiting**
   - LLM API calls (token budgets)
   - User upload limits
   - Concurrent request handling

4. **Async Processing**
   - Queue system for video creation
   - Background job processing (Celery, Bull)

---

## Deployment Checklist

- [ ] Frontend deployed to Vercel
- [ ] Backend API on Cloud Run/AWS Lambda
- [ ] ML model service on Hugging Face Spaces
- [ ] Database on Supabase/Railway
- [ ] Redis cache setup
- [ ] CDN configured for media
- [ ] Environment variables set
- [ ] SSL certificates configured
- [ ] Monitoring (Sentry, LogRocket)
- [ ] Analytics (PostHog, Mixpanel)

---

## Future Enhancements

1. Multi-language support (translate captions)
2. Hashtag suggestions (trending, niche-specific)
3. A/B testing for captions (performance metrics)
4. Batch processing (upload folder, generate all)
5. Browser extension (right-click image → generate caption)
6. Mobile app (iOS/Android)
7. Social media direct posting integration
8. AI-powered image editing suggestions
9. Voice narration for videos
10. Collaborative features (team workspaces)

---

## Resources & References

### Datasets
- MS COCO: https://cocodataset.org/
- Flickr30k: http://shannon.cs.illinois.edu/DenotationGraph/
- Conceptual Captions: https://ai.google.com/research/ConceptualCaptions/

### Pre-trained Models
- BLIP-2: https://huggingface.co/Salesforce/blip2-opt-2.7b
- CLIP: https://huggingface.co/openai/clip-vit-base-patch32
- YOLOv8: https://github.com/ultralytics/ultralytics

### API Documentation
- Gemini API: https://ai.google.dev/docs
- OpenAI API: https://platform.openai.com/docs
- Anthropic Claude: https://docs.anthropic.com/

### Tools
- FFmpeg: https://ffmpeg.org/
- Remotion: https://www.remotion.dev/
- Hugging Face: https://huggingface.co/

---

## Getting Started Commands

```bash
# Clone and setup
git clone <your-repo>
cd ai-caption-generator

# Frontend
cd frontend
npm install
npm run dev  # http://localhost:3000

# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload  # http://localhost:8000

# ML Model Service
cd ml-model
pip install -r requirements.txt
python inference/inference_engine.py  # http://localhost:8001

# Database
docker-compose up -d  # Starts PostgreSQL + Redis
```

---

## Success Metrics

- **Generation Speed**: < 5 seconds per caption
- **User Satisfaction**: 80%+ selection rate for generated options
- **Caption Quality**: 4+ star rating from users
- **System Uptime**: 99.9%
- **API Response Time**: < 2 seconds (P95)

---

This document provides the complete context for building the AI Caption Generator. Follow the phases sequentially, and adjust based on your team's expertise and timeline.
