import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import Modal from "./Modal";

// 与后端 diffusion_workbench_api/llm.py 的 _parse_content 同逻辑:
// 从模型输出中拆出 Positive/Negative 两段,容忍大小写和 markdown 噪声
const MARKER = /^[#\s\-*]*(positive|negative)\s*[:：]\s*/i;

function parsePromptPair(content) {
  const sections = {};
  let current = null;
  for (const rawLine of content.split("\n")) {
    const line = rawLine.replace(/\r$/, "");
    const match = line.match(MARKER);
    if (match) {
      current = match[1].toLowerCase();
      sections[current] = sections[current] || [];
      const rest = line.slice(match[0].length).trim();
      if (rest) sections[current].push(rest);
    } else if (current !== null) {
      sections[current].push(line.replace(/\s+$/, ""));
    }
  }
  if (!("negative" in sections)) return null;
  return {
    positive: (sections.positive || []).join("\n").trim(),
    negative: (sections.negative || []).join("\n").trim(),
  };
}

let nextMsgId = 1;

// 对话式快捷生成提示词弹窗:左 LLM 气泡(思考过程 + 流式输出),右用户气泡,
// 底部输入栏(语言/风格切换 + 发送)。历史为空即新对话,首条消息才创建 session。
// 组件常驻不卸载,open 仅控制显隐:关闭再打开仍保留上一次的会话内容。
export default function PromptAssistModal({ open, onClose, onUse }) {
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [input, setInput] = useState("");
  const [lang, setLang] = useState("en");
  const [promptStyle, setPromptStyle] = useState("flux");
  const [sending, setSending] = useState(false);
  const scrollRef = useRef(null);

  // 新消息与流式增量时保持滚动到底部
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  function newSession() {
    // 仅清空弹窗内容;新的 session id 在下一次发送首条消息时才创建
    setMessages([]);
    setSessionId(null);
    setInput("");
  }

  function changeContext(setter, value, currentValue) {
    if (value === currentValue) return;
    setter(value);
    // 语言或风格属于 system prompt 上下文,切换后从新会话开始。
    setMessages([]);
    setSessionId(null);
  }

  async function send() {
    const instruction = input.trim();
    if (!instruction || sending) return;
    const userMsg = { id: nextMsgId++, role: "user", content: instruction };
    const assistantMsg = {
      id: nextMsgId++,
      role: "assistant",
      content: "",
      thinking: "",
      status: "streaming",
    };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setInput("");
    setSending(true);

    const patchAssistant = (patch) =>
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantMsg.id ? { ...m, ...patch } : m)),
      );
    const appendAssistant = (key, delta) =>
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsg.id ? { ...m, [key]: m[key] + delta } : m,
        ),
      );

    try {
      const response = await api.promptAssistChat({
        instruction,
        language: lang,
        prompt_style: promptStyle,
        session_id: sessionId,
      });
      if (!response.ok) {
        // 400=未配置大模型;开始流式输出前的错误仍是普通 JSON 响应
        let detail = `HTTP ${response.status}`;
        try {
          const body = await response.json();
          if (body.detail) detail = String(body.detail);
        } catch {
          /* 保留默认错误信息 */
        }
        throw new Error(detail);
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let sep;
        while ((sep = buffer.indexOf("\n\n")) !== -1) {
          const rawEvent = buffer.slice(0, sep);
          buffer = buffer.slice(sep + 2);
          const dataLine = rawEvent
            .split("\n")
            .find((line) => line.startsWith("data:"));
          if (!dataLine) continue;
          let event;
          try {
            event = JSON.parse(dataLine.slice(5));
          } catch {
            continue;
          }
          if (event.type === "session") setSessionId(event.session_id);
          else if (event.type === "thinking") appendAssistant("thinking", event.delta);
          else if (event.type === "content") appendAssistant("content", event.delta);
          else if (event.type === "done") patchAssistant({ status: "done" });
          else if (event.type === "error") {
            patchAssistant({ status: "error", content: event.detail });
          }
        }
      }
      // 流被中断(未收到 done/error):有内容按完成处理,否则标记错误
      setMessages((prev) =>
        prev.map((m) => {
          if (m.id !== assistantMsg.id || m.status !== "streaming") return m;
          return m.content
            ? { ...m, status: "done" }
            : { ...m, status: "error", content: "连接中断,未收到完整回复" };
        }),
      );
    } catch (err) {
      patchAssistant({ status: "error", content: err.message });
    } finally {
      setSending(false);
    }
  }

  // 关闭时不渲染弹窗本体,但组件状态(消息、session id)保留
  if (!open) return null;

  return (
    <Modal
      id="prompt-assist-modal"
      title="快捷生成提示词"
      titleExtra={
        <button
          id="prompt-assist-new-session"
          type="button"
          className="prompt-assist-btn"
          title="新建会话"
          aria-label="新建会话"
          disabled={sending}
          onClick={newSession}
        >
          ＋
        </button>
      }
      onClose={onClose}
      closeDisabled={sending}
    >
      <div className="chat-messages" id="prompt-assist-messages" ref={scrollRef}>
        {messages.length === 0 && (
          <p className="chat-empty">
            描述想要的画面,帮你生成提示词;同一会话内可继续对话调整。
          </p>
        )}
        {messages.map((msg) =>
          msg.role === "user" ? (
            <div key={msg.id} className="chat-row chat-row-user">
              <div className="chat-bubble chat-bubble-user">{msg.content}</div>
            </div>
          ) : (
            <AssistantBubble key={msg.id} msg={msg} onUse={onUse} />
          ),
        )}
      </div>
      <div className="chat-input-bar" id="prompt-assist-input-bar">
        <textarea
          id="prompt-assist-input"
          value={input}
          placeholder={
            sessionId ? "继续对话调整提示词……" : "简单描述想要的画面……"
          }
          rows={2}
          autoFocus
          disabled={sending}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              send();
            }
          }}
        />
        <div className="chat-input-footer">
          <div className="chat-prompt-options">
            <OptionToggle
              label="输出语言"
              options={[
                { value: "en", label: "ENG" },
                { value: "zh", label: "中文" },
              ]}
              value={lang}
              disabled={sending}
              idPrefix="prompt-assist-lang"
              onChange={(value) => changeContext(setLang, value, lang)}
            />
            <OptionToggle
              label="提示词风格"
              options={[
                { value: "sd", label: "SD" },
                { value: "flux", label: "FLUX" },
              ]}
              value={promptStyle}
              disabled={sending}
              idPrefix="prompt-assist-style"
              onChange={(value) =>
                changeContext(setPromptStyle, value, promptStyle)
              }
            />
          </div>
          <button
            id="prompt-assist-send"
            type="button"
            className="primary chat-send-btn"
            disabled={sending || !input.trim()}
            onClick={send}
          >
            {sending ? "生成中…" : "发送"}
          </button>
        </div>
      </div>
    </Modal>
  );
}

function OptionToggle({ label, options, value, disabled, idPrefix, onChange }) {
  return (
    <div className="chat-option-toggle" role="group" aria-label={label}>
      {options.map((option) => (
        <button
          key={option.value}
          id={`${idPrefix}-${option.value}`}
          type="button"
          aria-pressed={value === option.value}
          className={value === option.value ? "active" : ""}
          disabled={disabled}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

function AssistantBubble({ msg, onUse }) {
  const parsed = msg.status === "done" ? parsePromptPair(msg.content) : null;
  return (
    <div className="chat-row chat-row-assistant">
      <div
        className={`chat-bubble chat-bubble-assistant${msg.status === "error" ? " error" : ""}`}
      >
        {msg.thinking && (
          <details className="chat-thinking" open={msg.status === "streaming"}>
            <summary>思考过程</summary>
            <div className="chat-thinking-body">{msg.thinking}</div>
          </details>
        )}
        {msg.status === "error" ? (
          <div className="chat-error">{msg.content}</div>
        ) : parsed ? (
          <>
            <div className="chat-section-label">Positive</div>
            <div className="chat-text">{parsed.positive}</div>
            <hr className="chat-divider" />
            <div className="chat-section-label">Negative</div>
            <div className="chat-text">{parsed.negative}</div>
          </>
        ) : (
          <div className="chat-text">
            {msg.content}
            {msg.status === "streaming" && <span className="chat-cursor">▍</span>}
          </div>
        )}
      </div>
      {msg.status === "done" && (
        <div className="chat-bubble-footer">
          <button
            type="button"
            className="chip chat-use-btn"
            disabled={!parsed}
            onClick={() => parsed && onUse(parsed.positive, parsed.negative)}
          >
            使用
          </button>
          {!parsed && (
            <span
              className="chat-warn"
              title="返回内容未按 Positive/Negative 格式输出,无法直接填入"
            >
              ⚠
            </span>
          )}
        </div>
      )}
    </div>
  );
}
