// Chainlit CustomElement: Mermaid 다이어그램 렌더러.
// CustomElement 런타임은 react/shadcn/lucide만 import 허용 → mermaid는 <script>로 주입한다.
// 로드 순서: 이미지에 번들된 /public/vendor (오프라인 서버) → jsDelivr CDN (로컬 개발).
import { useEffect, useRef, useState } from "react";

const SOURCES = [
  "/public/vendor/mermaid.min.js",
  "https://cdn.jsdelivr.net/npm/mermaid@12.1.0/dist/mermaid.min.js",
];

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.async = true;
    s.onload = () => (window.mermaid ? resolve(window.mermaid) : reject(new Error("mermaid global missing")));
    s.onerror = () => {
      s.remove();
      reject(new Error("failed to load " + src));
    };
    document.head.appendChild(s);
  });
}

function loadMermaid() {
  if (window.mermaid) return Promise.resolve(window.mermaid);
  if (!window.__myFriendMermaid) {
    // 여러 다이어그램이 동시에 마운트돼도 스크립트는 한 번만 로드
    window.__myFriendMermaid = SOURCES.reduce(
      (p, src) => p.catch(() => loadScript(src)),
      Promise.reject(new Error("init"))
    ).catch((e) => {
      window.__myFriendMermaid = null; // 실패는 캐시하지 않음 → 다음 렌더에서 재시도
      throw e;
    });
  }
  return window.__myFriendMermaid;
}

const isDark = () => document.documentElement.classList.contains("dark");

// Chainlit은 테마 토글 시 <html>의 class(light/dark)만 바꾼다 → SVG는 렌더 시점 색이 박혀 있으므로 다시 그려야 한다
function useDarkClass() {
  const [dark, setDark] = useState(isDark);
  useEffect(() => {
    const obs = new MutationObserver(() => setDark(isDark()));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => obs.disconnect();
  }, []);
  return dark;
}

export default function Mermaid() {
  const ref = useRef(null);
  const [error, setError] = useState(null);
  const [showSource, setShowSource] = useState(false);
  const dark = useDarkClass();

  useEffect(() => {
    let cancelled = false;
    setError(null);
    loadMermaid()
      .then(async (mermaid) => {
        // strict: 라벨의 HTML/스크립트 이스케이프 (라벨은 LLM 생성 텍스트)
        // wrappingWidth: 기본 200px은 role 라벨을 좁고 긴 노드로 접는다 → 넓혀서 세로 길이를 줄인다
        // dark edgeLabelBackground: 기본 #585858은 엣지 라벨과 대비가 낮다 (빨강 1.5:1) → 진한 단색
        mermaid.initialize({
          startOnLoad: false, securityLevel: "strict", theme: dark ? "dark" : "default",
          flowchart: { wrappingWidth: 320 },
          themeVariables: dark ? { edgeLabelBackground: "#262626" } : {},
        });
        // 불일치 엣지 라벨(render.py의 linkStyle color:#b91c1c)은 다크 배경에서 어둡다 → #f87171 (5.5:1)
        const code = dark ? props.code.replace(/^(\s*linkStyle [^\n]*)color:#b91c1c/gm, "$1color:#f87171") : props.code;
        const id = "mf-mmd-" + Math.random().toString(36).slice(2);
        const { svg } = await mermaid.render(id, code);
        if (!cancelled && ref.current) ref.current.innerHTML = svg;
      })
      .catch((e) => {
        if (!cancelled) setError(String(e && e.message ? e.message : e));
      });
    return () => {
      cancelled = true;
    };
  }, [props.code, dark]);

  return (
    <div className="w-full flex flex-col gap-2">
      {props.title ? <div className="text-sm font-semibold">{props.title}</div> : null}
      {error ? (
        <div className="text-sm text-red-600">Mermaid 렌더링 실패: {error}</div>
      ) : (
        <div ref={ref} className="w-full overflow-x-auto" />
      )}
      <button
        className="self-start text-xs underline opacity-70"
        onClick={() => setShowSource(!showSource)}
      >
        {showSource ? "소스 숨기기" : "Mermaid 소스 보기"}
      </button>
      {showSource || error ? (
        <pre className="text-xs whitespace-pre overflow-x-auto p-2 rounded bg-muted">{props.code}</pre>
      ) : null}
    </div>
  );
}
