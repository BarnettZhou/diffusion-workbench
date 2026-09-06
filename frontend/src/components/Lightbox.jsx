import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import Modal from "./Modal";
import { useMessage } from "./Message";

const MODE_LABELS = {
  zib: "z-image-base (zib)",
  zit: "z-image-turbo (zit)",
  krea2: "krea2 (krea2)",
  sdxl: "stable-diffusion-xl (sdxl)",
};

const VIDEO_MODEL_LABELS = {
  "wan2.2-ti2v-5b": "Wan 2.2 TI2V-5B",
  "wan2.2-i2v-14b": "Wan 2.2 I2V-14B",
  "minimax-h3": "MiniMax H3",
  "minimax-h3-fl2va": "MiniMax H3 FL2VA",
  "minimax-h3-ref2va": "MiniMax H3 Ref2VA",
  "minimax-h3-turbo": "MiniMax H3 FL2VA Turbo",
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
  kind = "image",
  onClose,
  onPrev = null,
  onNext = null,
  onSendToWorkbench = null,
  onSendToVideo = null,
  onSendToEdit = null,
  onSendToCaption = null,
  onDelete = null,
}) {
  const [meta, setMeta] = useState(null);
  const [drawerOpen, setDrawerOpen] = useState(
    () => window.matchMedia(DRAWER_DEFAULT_OPEN).matches,
  );
  // 视频播放模式:false=单次播放(默认),true=循环播放;切换上一张/下一张时保留
  const [loopPlayback, setLoopPlayback] = useState(false);
  const loopRef = useRef(loopPlayback);
  loopRef.current = loopPlayback;

  // 切换循环/单次;开启循环时若视频已播完(点击与播放结束撞车)立即重播。
  // 必须在事件处理函数内直接调用:被动 effect 里 play() 可能因激活过期被拒绝
  function toggleLoopPlayback() {
    const next = !loopRef.current;
    setLoopPlayback(next);
    if (next) {
      const video = document.getElementById("lightbox-video");
      if (video?.ended) video.play();
    }
  }

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
      else if (event.key === "Escape") {
        // 抽屉内弹出的 modal(如 LoRA 封面选择)优先消费 Esc,避免把图片预览一起关掉
        if (!document.querySelector(".modal-overlay")) onClose();
      }
      else if (event.key === "l" || event.key === "L") toggleLoopPlayback();
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

  // 截取 <video> 当前帧为 jpeg Blob(视频"设为封面"用);未加载完成时抛错
  function captureFrame() {
    const video = document.getElementById("lightbox-video");
    if (!video || !video.videoWidth) throw new Error("视频尚未加载完成,请稍候再试");
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    return new Promise((resolve, reject) =>
      canvas.toBlob(
        (blob) => (blob ? resolve(blob) : reject(new Error("封面截取失败"))),
        "image/jpeg",
        0.9,
      ),
    );
  }

  // 关闭 lightbox 的唯一显式路径:左上角 ×、抽屉内的 ×、Esc 键。
  // 不再支持点遮罩/点 body/点图片关闭(避免误触;在网格里点开看大图时尤其容易误点空白处)

  return (
    <div
      id="lightbox-overlay"
      className={drawerOpen ? "drawer-open" : undefined}
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
        {kind === "video" ? (
          // 视频:页内弹窗 + 浏览器原生播放器;循环开关在左下角
          <video
            id="lightbox-video"
            src={imageUrl}
            controls
            autoPlay
            playsInline
            loop={loopPlayback}
            onClick={(e) => e.stopPropagation()}
          />
        ) : (
          <img
            id="lightbox-image"
            src={imageUrl}
            alt={view.prompt ?? ""}
            onClick={(e) => e.stopPropagation()}
          />
        )}
      </div>
      {/* 左上角关闭按钮:不再支持点遮罩/点 body 关闭,这是唯一的显式关闭入口之一 */}
      <button
        id="lightbox-close-btn"
        type="button"
        aria-label="关闭大图"
        title="关闭大图 (Esc)"
        onClick={onClose}
      >
        ×
      </button>
      {kind === "video" && (
        // 视频时把循环播放挪到左下角,避免与左上角关闭按钮重叠
        <button
          id="lightbox-loop-btn"
          type="button"
          aria-pressed={loopPlayback}
          aria-label={loopPlayback ? "循环播放(开)" : "循环播放(关)"}
          title={
            loopPlayback ? "关闭循环播放,改为单次播放 (L)" : "开启循环播放 (L)"
          }
          className={loopPlayback ? "active" : undefined}
          onClick={(e) => {
            e.stopPropagation();
            toggleLoopPlayback();
          }}
        >
          循环播放
        </button>
      )}
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
          kind={kind}
          captureFrame={kind === "video" ? captureFrame : null}
          onClose={() => setDrawerOpen(false)}
          onSendToWorkbench={
            meta && onSendToWorkbench ? () => onSendToWorkbench(meta) : null
          }
          onSendToVideo={onSendToVideo}
          onSendToEdit={onSendToEdit}
          onSendToCaption={onSendToCaption}
          onDelete={onDelete}
        />
      )}
    </div>
  );
}

// 右侧信息抽屉:文件信息 + 生成参数 + 提示词(带复制),底部为操作工具栏
function InfoDrawer({ view, meta, imageUrl, fileInfo, kind = "image", captureFrame = null, onClose, onSendToWorkbench, onSendToVideo, onSendToEdit, onSendToCaption, onDelete = null }) {
  const message = useMessage();
  const [menuOpen, setMenuOpen] = useState(false);
  const [settingCover, setSettingCover] = useState(false);
  const [sendingToVideo, setSendingToVideo] = useState(false);
  const [sendingToEdit, setSendingToEdit] = useState(null); // 正在发送的编辑槽位:single / scene / subject
  const [sendingToCaption, setSendingToCaption] = useState(false);
  // 设为 LoRA 封面:图片模式且生成参数带 LoRA 才可用;多个 LoRA 时弹窗选择
  const [settingLoraCover, setSettingLoraCover] = useState(null); // 正在设置的 LoRA 文件名
  const [loraPickerOpen, setLoraPickerOpen] = useState(false);
  const loraItems = kind === "image" ? (view.loras ?? []) : [];
  const canSetLoraCover = Boolean(view.mode && loraItems.length);
  // 设为封面需要元数据里的分类与模型文件名;视频封面取当前帧,设到视频模型卡片
  const canSetCover =
    kind === "video"
      ? Boolean(meta?.video_model && meta?.model_name && captureFrame)
      : Boolean(meta?.mode && meta?.model_name);

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

  // 图片:把当前图片字节上传到封面端点;视频:截取当前帧设到视频模型卡片
  async function handleSetCover() {
    if (!canSetCover || settingCover) return;
    setSettingCover(true);
    try {
      if (kind === "video") {
        const blob = await captureFrame();
        await api.uploadVideoModelCover(meta.video_model, meta.model_name, blob);
      } else {
        const response = await fetch(imageUrl);
        if (!response.ok) throw new Error(`图片读取失败(HTTP ${response.status})`);
        const blob = await response.blob();
        await api.uploadModelCover(meta.mode, meta.model_name, blob);
      }
      message.success(`已设为 ${meta.model_name} 的封面`);
    } catch (err) {
      message.error(`设置封面失败:${err.message}`);
    } finally {
      setSettingCover(false);
    }
  }

  // 把当前图片导入为 I2V 输入图片(服务端本地复制)并跳到视频生成 tab
  async function handleSendToVideo() {
    if (!onSendToVideo || sendingToVideo) return;
    setSendingToVideo(true);
    try {
      await onSendToVideo();
    } catch (err) {
      message.error(`发送到视频生成失败:${err.message}`);
    } finally {
      setSendingToVideo(false);
    }
  }

  // 把当前图片导入为编辑输入图片(服务端本地复制)并跳到图像编辑 tab;
  // slot 决定预填到单图编辑还是多图编辑的场景/主体槽位
  async function handleSendToEdit(slot) {
    if (!onSendToEdit || sendingToEdit) return;
    setSendingToEdit(slot);
    try {
      await onSendToEdit(slot);
    } catch (err) {
      message.error(`发送到图片编辑失败:${err.message}`);
    } finally {
      setSendingToEdit(null);
    }
  }

  // 把当前图片字节设为指定 LoRA 的封面(与设为模型封面同一条上传通道)
  async function handleSetLoraCover(lora) {
    if (!canSetLoraCover || settingLoraCover) return;
    setSettingLoraCover(lora.name);
    try {
      const response = await fetch(imageUrl);
      if (!response.ok) throw new Error(`图片读取失败(HTTP ${response.status})`);
      const blob = await response.blob();
      await api.uploadLoraCover(view.mode, lora.name, blob);
      message.success(`已设为 ${lora.name} 的封面`);
    } catch (err) {
      message.error(`设置 LoRA 封面失败:${err.message}`);
    } finally {
      setSettingLoraCover(null);
      setLoraPickerOpen(false);
    }
  }

  // 菜单入口:只有一个 LoRA 直接设置,多个则弹窗列出供选择
  function handleSetLoraCoverClick() {
    if (loraItems.length === 1) handleSetLoraCover(loraItems[0]);
    else setLoraPickerOpen(true);
  }

  // 把当前图片导入为反推输入图片(与编辑共用受控通道)并跳到图片反推 tab
  async function handleSendToCaption() {
    if (!onSendToCaption || sendingToCaption) return;
    setSendingToCaption(true);
    try {
      await onSendToCaption();
    } catch (err) {
      message.error(`发送到图片反推失败:${err.message}`);
    } finally {
      setSendingToCaption(false);
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

        {meta?.source === "comfyui" && (
          <div id="drawer-source-comfyui" className="drawer-source-badge">
            由 ComfyUI 生成
          </div>
        )}

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
          {view.text_encoder_name && (
            <DrawerRow label="Text Encoder" value={view.text_encoder_name} mono />
          )}
          {view.loras?.length > 0 && (
            <DrawerRow
              label="LoRA"
              value={view.loras
                // 强度为 1 时不附带强度标注;非 1 时用方括号,避免圆括号看着像文件名的一部分
                .map((item) =>
                  Number(item.strength) === 1
                    ? item.name
                    : `${item.name} [${item.strength}]`,
                )
                .join(", ")}
              mono
            />
          )}
          {view.sampler && (
            <DrawerRow label="采样器" value={view.sampler} />
          )}
          {view.scheduler && (
            <DrawerRow label="调度器" value={view.scheduler} />
          )}
          {view.cfg != null && <DrawerRow label="CFG" value={String(view.cfg)} />}
          {view.shift != null && <DrawerRow label="Shift" value={String(view.shift)} />}
          {view.latent_multiplier != null && (
            <DrawerRow label="Latent multiplier" value={String(view.latent_multiplier)} />
          )}
          {view.steps != null && <DrawerRow label="步数" value={String(view.steps)} />}
          {view.seed != null && <DrawerRow label="Seed" value={String(view.seed)} />}
          {view.video_model && (
            <DrawerRow label="分类" value={VIDEO_MODEL_LABELS[view.video_model] ?? view.video_model} />
          )}
          {view.generation_type && (
            <DrawerRow label="类型" value={view.generation_type === "r2v" ? "参考生视频" : view.generation_type === "i2v" ? "图生视频" : "文生视频"} />
          )}
          {view.duration_seconds != null && (
            <DrawerRow label="时长 / 帧率" value={`${view.duration_seconds}s / ${view.fps}fps`} />
          )}
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
              {canSetLoraCover && (
                <button
                  id="drawer-set-lora-cover"
                  type="button"
                  role="menuitem"
                  disabled={Boolean(settingLoraCover)}
                  title="把当前图片设为生成时所用 LoRA 的封面"
                  onClick={() => {
                    setMenuOpen(false);
                    handleSetLoraCoverClick();
                  }}
                >
                  {settingLoraCover ? "设置中……" : "设置为 LoRA 封面"}
                </button>
              )}
              {onSendToVideo && (
                <button
                  id="drawer-send-to-video"
                  type="button"
                  role="menuitem"
                  disabled={sendingToVideo}
                  title="以此图片作为输入,生成图生视频(I2V)"
                  onClick={() => {
                    setMenuOpen(false);
                    handleSendToVideo();
                  }}
                >
                  {sendingToVideo ? "发送中……" : "发送到视频生成"}
                </button>
              )}
              {onSendToEdit && (
                <>
                  <button
                    id="drawer-send-to-edit-single"
                    type="button"
                    role="menuitem"
                    disabled={Boolean(sendingToEdit)}
                    title="以此图片作为单图编辑的输入"
                    onClick={() => {
                      setMenuOpen(false);
                      handleSendToEdit("single");
                    }}
                  >
                    {sendingToEdit === "single" ? "发送中……" : "发送到单图编辑"}
                  </button>
                  <button
                    id="drawer-send-to-edit-scene"
                    type="button"
                    role="menuitem"
                    disabled={Boolean(sendingToEdit)}
                    title="以此图片作为多图编辑的场景图(图片 1)"
                    onClick={() => {
                      setMenuOpen(false);
                      handleSendToEdit("scene");
                    }}
                  >
                    {sendingToEdit === "scene" ? "发送中……" : "发送到多图编辑-场景"}
                  </button>
                  <button
                    id="drawer-send-to-edit-subject"
                    type="button"
                    role="menuitem"
                    disabled={Boolean(sendingToEdit)}
                    title="以此图片作为多图编辑的主体图(图片 2)"
                    onClick={() => {
                      setMenuOpen(false);
                      handleSendToEdit("subject");
                    }}
                  >
                    {sendingToEdit === "subject" ? "发送中……" : "发送到多图编辑-主体"}
                  </button>
                </>
              )}
              {onSendToCaption && (
                <button
                  id="drawer-send-to-caption"
                  type="button"
                  role="menuitem"
                  disabled={sendingToCaption}
                  title="以此图片作为输入,进行图片反推"
                  onClick={() => {
                    setMenuOpen(false);
                    handleSendToCaption();
                  }}
                >
                  {sendingToCaption ? "发送中……" : "发送到图片反推"}
                </button>
              )}
              {/* 父级未注入 onDelete 时(ResultGallery 的 jobs 输出图)不显示,
                  相册注入后会在末尾追加删除项,与发送类操作隔开 */}
              {onDelete && <hr />}
              {onDelete && (
                <button
                  id="drawer-delete-image"
                  type="button"
                  role="menuitem"
                  className="danger-item"
                  title={`删除当前${kind === "video" ? "视频" : "图片"},将从本机移除文件`}
                  onClick={() => {
                    setMenuOpen(false);
                    onDelete();
                  }}
                >
                  {kind === "video" ? "删除视频" : "删除图片"}
                </button>
              )}
            </div>
          )}
        </div>
      </footer>

      {loraPickerOpen && (
        <Modal
          id="lora-cover-picker"
          title="设置为 LoRA 封面"
          onClose={() => setLoraPickerOpen(false)}
        >
          <div className="lora-cover-picker">
            {loraItems.map((item) => (
              <button
                key={item.name}
                id={`lora-cover-pick-${item.name}`}
                type="button"
                className="chip lora-cover-picker-item"
                disabled={Boolean(settingLoraCover)}
                onClick={() => handleSetLoraCover(item)}
              >
                {settingLoraCover === item.name
                  ? "设置中……"
                  : Number(item.strength) === 1
                    ? item.name
                    : `${item.name} [${item.strength}]`}
              </button>
            ))}
          </div>
        </Modal>
      )}
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
