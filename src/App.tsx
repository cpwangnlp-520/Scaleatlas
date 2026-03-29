import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom';
import { InferencePlanner } from './components/infer/InferencePlanner';
import { TrainPlanner } from './components/train/TrainPlanner';
import { APP_CHROME_COPY } from './content/plannerCopy.ts';
import { ParameterPage } from './pages';
import { usePlannerStore } from './stores';
import type { Locale } from './types';

function Navigation() {
  const locale = usePlannerStore((state) => state.locale);
  const setLocale = usePlannerStore((state) => state.setLocale);
  const copy = APP_CHROME_COPY[locale];

  return (
    <header className="workspace-topbar">
      <div className="workspace-toolbar workspace-toolbar-minimal">
        <NavLink to="/" className="workspace-brand workspace-brand-minimal">
          <div className="workspace-brand-title">{copy.brandTitle}</div>
        </NavLink>

        <nav className="workspace-nav" aria-label={locale === 'zh' ? '主导航' : 'Primary'}>
          <div className="workspace-nav-rail">
            {copy.nav.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) => `workspace-nav-pill tone-${item.tone}${isActive ? ' active' : ''}`}
              >
                {item.label}
              </NavLink>
            ))}
          </div>
        </nav>

        <div className="workspace-locale-rail">
          {(['zh', 'en'] as Locale[]).map((nextLocale) => (
            <button
              key={nextLocale}
              type="button"
              className={`workspace-locale-pill${locale === nextLocale ? ' active' : ''}`}
              onClick={() => setLocale(nextLocale)}
            >
              {nextLocale === 'zh' ? 'ZH' : 'EN'}
            </button>
          ))}
        </div>
      </div>
    </header>
  );
}

function AppLayout() {
  return (
    <div className="min-h-screen">
      <div className="workspace-shell">
        <Navigation />
        <main className="workspace-main">
          <Routes>
            <Route path="/" element={<ParameterPage />} />
            <Route path="/train" element={<TrainPlanner />} />
            <Route path="/infer" element={<InferencePlanner />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AppLayout />
    </BrowserRouter>
  );
}

export default App;
