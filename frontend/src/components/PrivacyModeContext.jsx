import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

// 隐私模式:只影响当前前端访问(不写入服务端),用 sessionStorage 在
// 同一浏览器 tab 内跨刷新保留;关闭 tab / 新开 tab 自动回到关闭状态。
// 开启后:
//   1. 工作台模型选择器的封面图片替换为「隐私模式」占位
//   2. 设置页「生图模型」「视频模型」两个 tab 的封面替换为「隐私模式」占位
//   3. 顶部「相册」tab 不可点击(不会切换 tab,也不会拉取相册内容)
const STORAGE_KEY = "dwb.privacyMode";
const PrivacyModeContext = createContext(null);

function readInitial() {
  try {
    return sessionStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

export function PrivacyModeProvider({ children }) {
  const [privacyMode, setPrivacyMode] = useState(readInitial);

  useEffect(() => {
    try {
      if (privacyMode) sessionStorage.setItem(STORAGE_KEY, "1");
      else sessionStorage.removeItem(STORAGE_KEY);
    } catch {
      /* sessionStorage 不可用时静默降级为纯内存 */
    }
  }, [privacyMode]);

  const toggle = useCallback(() => setPrivacyMode((prev) => !prev), []);

  const value = useMemo(
    () => ({ privacyMode, setPrivacyMode, toggle }),
    [privacyMode, toggle],
  );

  return (
    <PrivacyModeContext.Provider value={value}>
      {children}
    </PrivacyModeContext.Provider>
  );
}

export function usePrivacyMode() {
  const ctx = useContext(PrivacyModeContext);
  if (!ctx) {
    throw new Error("usePrivacyMode must be used within PrivacyModeProvider");
  }
  return ctx;
}
