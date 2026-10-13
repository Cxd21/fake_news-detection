# Frontend — Get Legit Intelligence

React 19 + Vite 8 + Tailwind CSS v4 single-page application.

## Directory Structure

To maintain compatibility with Figma Make's live preview server and hot-reloading (which watches the repository root), the primary frontend source files reside in the root directory:

- `src/App.tsx` — Main application component (Analyzer, Accuracy transparency page, client-side routing)
- `src/main.tsx` — React entrypoint mounting `#root`
- `src/index.css` — Global styles and Tailwind CSS v4 imports
- `public/` — Static assets (logos, favicon, robots.txt)
- `index.html` — HTML shell
- `package.json` — Frontend dependencies and build scripts
- `vite.config.ts` — Vite build and dev proxy configuration (`/api` -> Flask on `http://127.0.0.1:5000`)
- `dist/` — Production build output served directly by Flask

## Scripts

```bash
# Start development server with hot-reload
npm run dev

# Build for production (outputs to dist/)
npm run build

# Preview production build
npm run preview
```

