import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import Modal from "./Modal";
import { useMessage } from "./Message";

const PAGE_SIZE = 10;
const PREVIEW_LEN = 60;

// 请求/返回内容在表格里只展示单行摘要,点击后弹窗看全部
function summarize(text) {
  const flat = String(text ?? "").replace(/\s+/g, " ").trim();
  if (!flat) return "—";
  return flat.length > PREVIEW_LEN ? `${flat.slice(0, PREVIEW_LEN)}…` : flat;
}

function tokenText(value) {
  return value === null || value === undefined ? "—" : String(value);
}

export default function LLMRecords() {
  const message = useMessage();
  const [page, setPage] = useState(1);
  const [data, setData] = useState({ total: 0, items: [] });
  const [loading, setLoading] = useState(false);
  const [detail, setDetail] = useState(null); // { title, content }

  const load = useCallback(
    async (targetPage) => {
      setLoading(true);
      try {
        const result = await api.llmRequests(targetPage, PAGE_SIZE);
        // 记录被裁剪后页码可能越界,回退到最后一页
        const maxPage = Math.max(1, Math.ceil(result.total / PAGE_SIZE));
        if (targetPage > maxPage) {
          setPage(maxPage);
          return;
        }
        setData(result);
      } catch (err) {
        message.error(`加载请求记录失败:${err.message}`);
      } finally {
        setLoading(false);
      }
    },
    [message],
  );

  useEffect(() => {
    load(page);
  }, [page, load]);

  const totalPages = Math.max(1, Math.ceil(data.total / PAGE_SIZE));

  return (
    <div id="settings-llm-records" className="panel">
      <h2>
        请求记录
        <button
          id="llm-records-refresh"
          type="button"
          className="chip"
          disabled={loading}
          onClick={() => load(page)}
        >
          {loading ? "加载中……" : "刷新"}
        </button>
      </h2>
      <p className="settings-item-desc">
        每次大模型调用使用独立 session;最多保留最近 200 条记录。
      </p>
      {data.items.length === 0 ? (
        <p className="settings-placeholder" id="llm-records-empty">
          暂无请求记录
        </p>
      ) : (
        <table id="llm-records-table">
          <thead>
            <tr>
              <th>请求时间</th>
              <th>Base URL</th>
              <th>模型 ID</th>
              <th>Session ID</th>
              <th>请求内容</th>
              <th>返回内容</th>
              <th>输入 tokens</th>
              <th>输出 tokens</th>
              <th>缓存命中</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map((item) => (
              <tr key={item.id} id={`llm-record-row-${item.id}`}>
                <td className="llm-record-time">{item.timestamp}</td>
                <td className="llm-record-url" title={item.base_url}>
                  {item.base_url}
                </td>
                <td>{item.model}</td>
                <td className="llm-record-session" title={item.session_id}>
                  {item.session_id.slice(0, 8)}
                </td>
                <td>
                  <button
                    type="button"
                    className="llm-record-cell"
                    id={`llm-record-request-${item.id}`}
                    onClick={() =>
                      setDetail({ title: "请求内容", content: item.request })
                    }
                  >
                    {summarize(item.request)}
                  </button>
                </td>
                <td>
                  <button
                    type="button"
                    className={`llm-record-cell${item.error ? " error" : ""}`}
                    id={`llm-record-response-${item.id}`}
                    onClick={() =>
                      setDetail({
                        title: item.error ? "错误信息" : "返回内容",
                        content: item.error || item.response,
                      })
                    }
                  >
                    {item.error ? summarize(item.error) : summarize(item.response)}
                  </button>
                </td>
                <td className="llm-record-num">{tokenText(item.input_tokens)}</td>
                <td className="llm-record-num">{tokenText(item.output_tokens)}</td>
                <td className="llm-record-num">{tokenText(item.cached_tokens)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {data.total > 0 && (
        <div id="llm-records-pager">
          <button
            id="llm-records-prev"
            type="button"
            className="chip"
            disabled={loading || page <= 1}
            onClick={() => setPage((p) => Math.max(1, p - 1))}
          >
            上一页
          </button>
          <span id="llm-records-page-info">
            第 {page} / {totalPages} 页 · 共 {data.total} 条
          </span>
          <button
            id="llm-records-next"
            type="button"
            className="chip"
            disabled={loading || page >= totalPages}
            onClick={() => setPage((p) => p + 1)}
          >
            下一页
          </button>
        </div>
      )}
      {detail && (
        <Modal
          id="llm-record-detail-modal"
          title={detail.title}
          onClose={() => setDetail(null)}
        >
          <pre className="llm-record-detail">{detail.content}</pre>
        </Modal>
      )}
    </div>
  );
}
