import { useEffect, useState } from "react";
import { api } from "../api/client";
import { useMessage } from "./Message";

const MODE_LABELS = {
  zib: "z-image-base (zib)",
  zit: "z-image-turbo (zit)",
  krea2: "krea2 (krea2)",
  sdxl: "stable-diffusion-xl (sdxl)",
};

// 宽屏(桌面)默认展开抽屉,窄屏(移动端)默认收起,由右上角悬浮按钮切换
const DRAWER_DEFAULT_OPEN = "(min-width: 901px)";

// 通用全屏图片预览:右侧抽屉展示文件信息与 PNG 内嵌生成参数,
// 字段缺失时不渲染对应行;fileInfo 可选(相册提供位置/大小/时间)。
export default function Lightbox({
  imageUrl,
  metadataUrl,
  fallback,
  fileInfo = null,
  onClose,
  onPrev = null,
  onNext = null,
  onSendToWorkbench = null,
}) {
  const [meta, setMeta] = useState(null);
  const [drawerOpen, setDrawerOpen] = useState(
    () => window.matchMedia(DRAWER_DEFAULT_OPEN).matches,
  );

  // 预览打开时锁定背景页面滚动
  useEffect(() => {
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, []);

  // 键盘导航:← 上一张,→ 下一张,Esc 关闭
  useEffect(() => {
    function onKeyDown(event) {
      if (event.key === "ArrowLeft") onPrev?.();
      else if (event.key === "ArrowRight") onNext?.();
      else if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onPrev, onNext, onClose]);

  useEffect(() => {
    setMeta(null); // 切换图片时清除上一张的元数据
    if (!metadataUrl) return undefined;
    let cancelled = false;
    api
      .get(metadataUrl)
      .then((data) => !cancelled && setMeta(data))
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [metadataUrl]);

  const view = meta ?? fallback ?? {};

  return (
    <div
      id="lightbox-overlay"
      className={drawerOpen ? "drawer-open" : undefined}
      onClick={onClose}
    >
      {onPrev && (
        <button
          id="lightbox-prev"
          className="lightbox-nav prev"
          type="button"
          aria-label="上一张"
          onClick={(e) => {
            e.stopPropagation();
            onPrev();
          }}
        >
          ‹
        </button>
      )}
      {onNext && (
        <button
          id="lightbox-next"
          className="lightbox-nav next"
          type="button"
          aria-label="下一张"
          onClick={(e) => {
            e.stopPropagation();
            onNext();
          }}
        >
          ›
        </button>
      )}
      <div className="lightbox-body">
        <img
          id="lightbox-image"
          src={imageUrl}
          alt={view.prompt ?? ""}
          onClick={(e) => e.stopPropagation()}
        />
      </div>
      <button
        id="lightbox-info-btn"
        type="button"
        aria-label={drawerOpen ? "收起图片信息" : "展开图片信息"}
        title={drawerOpen ? "收起图片信息" : "展开图片信息"}
        onClick={(e) => {
          e.stopPropagation();
          setDrawerOpen((open) => !open);
        }}
      >
        ⓘ
      </button>
      {drawerOpen && (
        <InfoDrawer
          view={view}
          meta={meta}
          imageUrl={imageUrl}
          fileInfo={fileInfo}
          onClose={() => setDrawerOpen(false)}
          onSendToWorkbench={
            meta && onSendToWorkbench ? () => onSendToWorkbench(meta) : null
          }
        />
      )}
    </div>
  );
}

// 右侧信息抽屉:文件信息 + 生成参数 + 提示词(带复制),底部为操作工具栏
function InfoDrawer({ view, meta, imageUrl, fileInfo, onClose, onSendToWorkbench }) {
  const message = useMessage();
  const [menuOpen, setMenuOpen] = useState(false);
  const [settingCover, setSettingCover] = useState(false);
  // 设为封面需要元数据里的 mode 和模型文件名
  const canSetCover = Boolean(meta?.mode && meta?.model_name);

  // 尺寸以当前文件为准:artifact 是 PNG 实际尺寸(放大图是放大后的尺寸),
  // width/height 是首次生成尺寸,旧 schema 无 artifact 时回退
  const fileWidth = meta?.artifact?.width ?? meta?.width ?? view.width;
  const fileHeight = meta?.artifact?.height ?? meta?.height ?? view.height;

  // dropup 打开后,点击任意处 / Esc 关闭
  useEffect(() => {
    if (!menuOpen) return undefined;
    const close = () => setMenuOpen(false);
    const onKey = (event) => event.key === "Escape" && close();
    window.addEventListener("click", close);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("click", close);
      window.removeEventListener("keydown", onKey);
    };
  }, [menuOpen]);

  // 把当前图片字节上传到封面端点,服务端保存到 .cache/covers/{mode}/
  async function handleSetCover() {
    if (!canSetCover || settingCover) return;
    setSettingCover(true);
    try {
      const response = await fetch(imageUrl);
      if (!response.ok) throw new Error(`图片读取失败(HTTP ${response.status})`);
      const blob = await response.blob();
      await api.uploadModelCover(meta.mode, meta.model_name, blob);
      message.success(`已设为 ${meta.model_name} 的封面`);
    } catch (err) {
      message.error(`设置封面失败:${err.message}`);
    } finally {
      setSettingCover(false);
    }
  }

  return (
    <aside
      id="lightbox-drawer"
      aria-label="图片信息"
      onClick={(e) => e.stopPropagation()}
    >
      <div className="drawer-scroll">
        <button
          id="drawer-close-btn"
          type="button"
          aria-label="收起"
          onClick={onClose}
        >
          ×
        </button>

        {fileInfo && (
          <section className="drawer-section">
            <h3>文件</h3>
            <DrawerRow label="位置" value={fileInfo.location} mono />
            {fileInfo.sizeBytes != null && (
              <DrawerRow
                label="大小"
                value={`${(fileInfo.sizeBytes / 1048576).toFixed(2)} MB`}
              />
            )}
            {fileWidth && fileHeight && (
              <DrawerRow label="尺寸" value={`${fileWidth}×${fileHeight}`} />
            )}
            {fileInfo.mtimeNs != null && (
              <DrawerRow label="生成时间" value={formatTime(fileInfo.mtimeNs)} />
            )}
          </section>
        )}

        <section className="drawer-section">
          <h3>生成参数</h3>
          {view.mode && (
            <DrawerRow label="模式" value={MODE_LABELS[view.mode] ?? view.mode} />
          )}
          {view.model_name && <DrawerRow label="模型" value={view.model_name} mono />}
          {view.vae_name && <DrawerRow label="VAE" value={view.vae_name} mono />}
          {view.sampler && (
            <DrawerRow
              label="采样 / 调度"
              value={`${view.sampler} / ${view.scheduler}`}
            />
          )}
          {view.cfg != null && <DrawerRow label="CFG" value={String(view.cfg)} />}
          {view.steps != null && <DrawerRow label="步数" value={String(view.steps)} />}
          {view.seed != null && <DrawerRow label="Seed" value={String(view.seed)} />}
        </section>

        {view.prompt && (
          <PromptSection
            id="drawer-prompt"
            title="正向提示词"
            text={view.prompt}
          />
        )}
        {view.negative_prompt ? (
          <PromptSection
            id="drawer-negative-prompt"
            title="负面提示词"
            text={view.negative_prompt}
          />
        ) : null}
      </div>

      <footer className="drawer-toolbar">
        {onSendToWorkbench && (
          <button
            id="send-to-workbench-btn"
            type="button"
            className="primary"
            onClick={onSendToWorkbench}
          >
            发送到工作台
          </button>
        )}
        <div className="drawer-menu-wrap">
          <button
            id="drawer-menu-btn"
            type="button"
            className="drawer-menu-btn"
            aria-label="更多操作"
            title="更多操作"
            onClick={(e) => {
              e.stopPropagation();
              setMenuOpen((open) => !open);
            }}
          >
            ☰
          </button>
          {menuOpen && (
            <div className="drawer-dropup" role="menu">
              <button
                id="drawer-set-cover"
                type="button"
                role="menuitem"
                disabled={!canSetCover || settingCover}
                title={canSetCover ? undefined : "图片缺少模型信息"}
                onClick={() => {
                  setMenuOpen(false);
                  handleSetCover();
                }}
              >
                {settingCover ? "设置中……" : "设置为封面"}
              </button>
            </div>
          )}
        </div>
      </footer>
    </aside>
  );
}

function DrawerRow({ label, value, mono = false }) {
  return (
    <div className="drawer-row">
      <span className="drawer-label">{label}</span>
      <span className={`drawer-value${mono ? " mono" : ""}`} title={value}>
        {value}
      </span>
    </div>
  );
}

// 提示词小节:label 行右侧放复制按钮(复制失败时降级到 execCommand),正文卡片只放文本
function PromptSection({ id, title, text }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <section className="drawer-section">
      <div className="drawer-section-head">
        <h3>{title}</h3>
        <button
          type="button"
          className="drawer-copy-btn"
          onClick={copy}
        >
          {copied ? "已复制" : "复制"}
        </button>
      </div>
      <div className="drawer-prompt" id={id}>
        <div className="drawer-prompt-text">{text}</div>
      </div>
    </section>
  );
}

function formatTime(mtimeNs) {
  return new Date(mtimeNs / 1e6).toLocaleString("zh-CN", { hour12: false });
}
