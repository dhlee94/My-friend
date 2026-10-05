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

export default function Mermaid() {
  const ref = useRef(null);
  const [error, setError] = useState(null);
  const [showSource, setShowSource] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    loadMermaid()
      .then(async (mermaid) => {
        const dark = document.documentElement.classList.contains("dark");
        // strict: 라벨의 HTML/스크립트 이스케이프 (라벨은 LLM 생성 텍스트)
        mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: dark ? "dark" : "default" });
        const id = "mf-mmd-" + Math.random().toString(36).slice(2);
        const { svg } = await mermaid.render(id, props.code);
        if (!cancelled && ref.current) ref.current.innerHTML = svg;
      })
      .catch((e) => {
        if (!cancelled) setError(String(e && e.message ? e.message : e));
      });
    return () => {
      cancelled = true;
    };
  }, [props.code]);

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
