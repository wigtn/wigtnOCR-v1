# Spotfinder - Seongsu Popup Store Finder

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 15](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org/)
[![React Native](https://img.shields.io/badge/React%20Native-Expo-blue.svg)](https://expo.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AI-powered popup store discovery platform for Seongsu-dong, Seoul**

[한국어 문서](./README.ko.md)

Spotfinder helps tourists (especially Japanese visitors) discover trending popup stores in Seongsu-dong through an AI chat assistant. Built with a modern Monorepo architecture supporting web, mobile, and API services.

## Features

- **AI Chat Assistant**: Conversational popup store recommendations
- **Multilingual Support**: Japanese, English, Korean
- **Real-time Popup Data**: Instagram scraping with AI-powered parsing
- **Interactive Map**: Naver Map integration for navigation
- **Cross-platform**: Web (Next.js) + Mobile (React Native)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Client Applications                             │
├─────────────────────────────┬─────────────────────────┬─────────────────────┤
│         apps/web            │       apps/mobile       │    External Users   │
│     (Next.js 15 + RSC)      │   (React Native/Expo)   │                     │
│      Vercel Deployed        │    App Store / Play     │                     │
└─────────────────────────────┴─────────────────────────┴─────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              apps/api (FastAPI)                              │
│                             Railway Deployed                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  API Gateway    │  │  Business Agent │  │   Instagram Scraper         │  │
│  │  • /api/chat    │  │  • LangGraph    │  │   • Instaloader            │  │
│  │  • /api/popups  │  │  • Tool Calling │  │   • Upstage Document AI    │  │
│  │  • /health      │  │  • Streaming    │  │   • APScheduler            │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Data Layer                                      │
├──────────────────────┬──────────────────────┬───────────────────────────────┤
│       SQLite         │       Qdrant         │         Naver API             │
│   • Popup Store DB   │   • Vector Memory    │   • Map / Directions          │
│   • Local Cache      │   • Embeddings       │   • Geocoding                 │
└──────────────────────┴──────────────────────┴───────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Web Frontend** | Next.js 15, React 19, Tailwind CSS, Supabase Auth |
| **Mobile App** | React Native, Expo Router, Supabase |
| **Backend API** | FastAPI, LangGraph, LangChain |
| **LLM** | Upstage Solar / OpenAI (fallback) |
| **Database** | SQLite (popups), Qdrant (vectors) |
| **External APIs** | Naver Map, Naver Geocoding, Instagram |
| **Deployment** | Vercel (web), Railway (api), EAS (mobile) |

## Project Structure

```
spotfinder/
├── apps/
│   ├── api/                    # FastAPI Backend
│   │   ├── src/
│   │   │   ├── agents/         # LangGraph AI Agents
│   │   │   ├── api/            # FastAPI Routes & Middleware
│   │   │   ├── config/         # Settings (Pydantic)
│   │   │   ├── db/             # Database (SQLite, Qdrant)
│   │   │   ├── models/         # Domain Models
│   │   │   ├── scraper/        # Instagram Scraper
│   │   │   ├── services/       # LLM, Memory, Embeddings
│   │   │   └── tools/          # Naver API, Translation
│   │   ├── scripts/            # CLI utilities
│   │   ├── tests/              # pytest tests
│   │   ├── Dockerfile
│   │   └── pyproject.toml      # uv dependencies
│   │
│   ├── web/                    # Next.js Web App
│   │   ├── src/
│   │   │   ├── app/            # App Router pages
│   │   │   ├── components/     # UI components
│   │   │   ├── features/       # Feature modules
│   │   │   └── lib/            # Utilities
│   │   ├── Dockerfile
│   │   └── package.json
│   │
│   └── mobile/                 # React Native App
│       ├── app/                # Expo Router
│       ├── components/
│       └── package.json
│
├── docs/                       # Documentation
├── docker-compose.yml          # Local development
├── railway.toml                # Railway deployment
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.11+ & [uv](https://github.com/astral-sh/uv)
- Node.js 20+
- Docker (optional, for local services)

### Backend (API)

```bash
cd apps/api

# Install dependencies with uv
uv sync

# Set environment variables
cp ../../.env.example .env

# Run development server
uv run uvicorn src.api.main:app --reload --port 8080
```

### Frontend (Web)

```bash
cd apps/web

# Install dependencies
npm install

# Set environment variables
cp .env.example .env.local

# Run development server
npm run dev
```

### Mobile

```bash
cd apps/mobile

# Install dependencies
npm install

# Start Expo
npm start
```

### Docker Compose (Full Stack)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api
```

## Deployment

| Service | Platform | Config File |
|---------|----------|-------------|
| **API** | Railway | `railway.toml` |
| **Web** | Vercel | `apps/web/vercel.json` |
| **Mobile** | EAS Build | `apps/mobile/app.json` |

### Railway (API)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
railway up
```

### Vercel (Web)

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy from apps/web
cd apps/web && vercel
```

## API Endpoints

### Chat

```http
POST /api/v1/chat
Content-Type: application/json

{
  "message": "성수동 패션 팝업 추천해줘",
  "thread_id": "session-123"
}
```

### Popups

```http
GET /api/v1/popups?category=fashion&active_only=true
```

### Health

```http
GET /health
```

## Environment Variables

```env
# LLM
UPSTAGE_API_KEY=your-upstage-key
OPENAI_API_KEY=your-openai-key  # fallback

# Naver API
NAVER_CLIENT_ID=your-client-id
NAVER_CLIENT_SECRET=your-client-secret
NAVER_MAP_CLIENT_ID=your-map-client-id
NAVER_MAP_CLIENT_SECRET=your-map-secret

# Database
QDRANT_URL=http://localhost:6333

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8080
NEXT_PUBLIC_SUPABASE_URL=your-supabase-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-key
```

## Development

### Testing

```bash
cd apps/api

# Run all tests
uv run pytest

# With coverage
uv run pytest --cov=src
```

### Code Quality

```bash
# Lint & format
uv run ruff check src --fix
uv run ruff format src

# Type check
uv run mypy src
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) - AI Agent Framework
- [Upstage](https://www.upstage.ai/) - Solar LLM & Document AI
- [Naver Cloud Platform](https://www.ncloud.com/) - Map APIs
