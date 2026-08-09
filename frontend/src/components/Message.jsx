import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
} from "react";

// 全局轻提示:success / info / warning / error,自动消失,点击立即关闭。
// 用法:const message = useMessage(); message.success("已保存");
const MessageContext = createContext(null);

const DURATIONS = { success: 3000, info: 3000, warning: 4000, error: 5000 };

export function MessageProvider({ children }) {
  const [items, setItems] = useState([]);
  const idRef = useRef(0);
  const timers = useRef({});

  const dismiss = useCallback((id) => {
    setItems((prev) => prev.filter((item) => item.id !== id));
    clearTimeout(timers.current[id]);
    delete timers.current[id];
  }, []);

  const push = useCallback(
    (type, text) => {
      const id = ++idRef.current;
      // 同屏最多 5 条,超出丢弃最旧
      setItems((prev) => [...prev.slice(-4), { id, type, text }]);
      timers.current[id] = setTimeout(
        () => dismiss(id),
        DURATIONS[type] ?? 3000,
      );
    },
    [dismiss],
  );

  const api = useMemo(
    () => ({
      success: (text) => push("success", text),
      info: (text) => push("info", text),
      warning: (text) => push("warning", text),
      error: (text) => push("error", text),
    }),
    [push],
  );

  return (
    <MessageContext.Provider value={api}>
      {children}
      <div id="message-container" role="status" aria-live="polite">
        {items.map((item) => (
          <div
            key={item.id}
            className={`message-item message-${item.type}`}
            onClick={() => dismiss(item.id)}
          >
            {item.text}
          </div>
        ))}
      </div>
    </MessageContext.Provider>
  );
}

export function useMessage() {
  return useContext(MessageContext);
}
