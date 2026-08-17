import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import ContextMenu from "./ContextMenu";
import Lightbox from "./Lightbox";
import Modal from "./Modal";

const PAGE_SIZE = 60;
// 提前 600px 加载,移出范围后回收图片资源
const VIEWPORT_MARGIN = "600px";
const BUILTIN_DIR = "output";

// 相册:顶部 tabs 为目录列表(内置 output + 用户添加的目录),网格展示当前
// 目录全部图片,按修改时间倒序。数据来自后端扫盘索引(不依赖 jobs 表),
// 容忍图片被删除/移动。
export default function AlbumPage({ onSendToWorkbench, onSendToVideo, onSendVideoMeta }) {
  const [dirs, setDirs] = useState(null);
  const [activeDir, setActiveDir] = useState(BUILTIN_DIR);
  const [images, setImages] = useState([]);
  // 当前 images 列表所属的目录;切目录后的旧列表不再渲染,避免旧 relpath 配新 dir 请求 404
  const [imagesDir, setImagesDir] = useState(BUILTIN_DIR);
  const [cursor, setCursor] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);
  // 子目录浏览:当前查看的一级子目录(null=查看全部);汉堡弹层开关与目录列表
  const [subDir, setSubDir] = useState(null);
  const [subPickerOpen, setSubPickerOpen] = useState(false);
  const [subdirs, setSubdirs] = useState(null);
  // 当前 images 列表所属的子目录,与 imagesDir 一起防止旧列表配新范围渲染
  const [imagesSubdir, setImagesSubdir] = useState(null);
  // 图片右键菜单:{x, y, image};待确认删除的图片;删除中状态
  const [menu, setMenu] = useState(null);
  const [pendingDelete, setPendingDelete] = useState(null);
  const [deleting, setDeleting] = useState(false);
  // 批量选择:开关、已选图片 id 集合、待确认批量删除、删除中状态
  const [batchMode, setBatchMode] = useState(false);
  const [batchSelected, setBatchSelected] = useState(() => new Set());
  const [pendingBatchDelete, setPendingBatchDelete] = useState(false);
  const [batchDeleting, setBatchDeleting] = useState(false);
  // 目录 tabs:add/edit modal 开关与编辑对象;tab 右键菜单;待确认删除的目录
  const [dirModal, setDirModal] = useState(null); // {mode: "add" | "edit", dir?}
  const [dirMenu, setDirMenu] = useState(null); // {x, y, dir}
  const [pendingDeleteDir, setPendingDeleteDir] = useState(null);
  const sentinelRef = useRef(null);
  const subPanelRef = useRef(null);
  const stateRef = useRef({ cursor: null, loading: false, dir: BUILTIN_DIR, subdir: null });
  // 已删除文件的 id(relpath)集合:阻止删除前发出的翻页请求把已删项重新带回列表
  const removedIdsRef = useRef(new Set());
  stateRef.current = { cursor, loading, dir: activeDir, subdir: subDir };

  const loadMore = useCallback(async () => {
    const { cursor: current, loading: busy, dir, subdir } = stateRef.current;
    if (busy || current === "end") return;
    setLoading(true);
    try {
      const page = await api.album(PAGE_SIZE, current, dir, subdir);
      // 请求发出后目录/子目录已切换:丢弃过期页,loading 交给新范围的请求收尾
      if (stateRef.current.dir !== dir || stateRef.current.subdir !== subdir) return;
      // 页面可能扫描于删除之前,含已删项;追加时按 removedIdsRef 过滤
      setImages((prev) => [
        ...prev,
        ...page.images.filter((item) => !removedIdsRef.current.has(item.id)),
      ]);
      setImagesDir(dir);
      setImagesSubdir(subdir);
      setCursor(page.next_cursor ?? "end");
    } catch (err) {
      if (stateRef.current.dir === dir && stateRef.current.subdir === subdir)
        setError(err.message);
    } finally {
      if (stateRef.current.dir === dir && stateRef.current.subdir === subdir)
        setLoading(false);
    }
  }, []);

  // 目录列表加载
  useEffect(() => {
    api
      .albumDirs()
      .then((data) => setDirs(data.dirs))
      .catch((err) => setError(`目录列表加载失败:${err.message}`));
  }, []);

  // 重新加载当前目录(切目录首载与手动刷新共用):清空网格与现场,从头加载
  const reload = useCallback(() => {
    setImages([]);
    setCursor(null);
    setSelected(null);
    setMenu(null);
    setError(null);
    setBatchMode(false);
    setBatchSelected(new Set());
    removedIdsRef.current = new Set();
    stateRef.current = { cursor: null, loading: false, dir: activeDir, subdir: subDir };
    loadMore();
  }, [activeDir, subDir, loadMore]);

  // 切换目录时重新加载
  useEffect(() => {
    reload();
  }, [reload]);

  // 滚动到底部附近时翻页(无限滚动)
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => entries[0].isIntersecting && loadMore(),
      { rootMargin: "800px" },
    );
    if (sentinelRef.current) observer.observe(sentinelRef.current);
    return () => observer.disconnect();
  }, [loadMore]);

  // 打开子目录弹层时,把当前选中的子目录滚动到弹层可视位置
  useEffect(() => {
    if (!subPickerOpen) return;
    const panel = subPanelRef.current;
    const activeButton = panel?.querySelector("button.active");
    if (panel && activeButton) {
      panel.scrollTop =
        activeButton.offsetTop - panel.clientHeight / 2 + activeButton.clientHeight / 2;
    }
  }, [subPickerOpen, subdirs]);

  // 浏览到最后一张已加载图片时提前翻页,保证 lightbox 内可继续向后翻
  useEffect(() => {
    if (selected && images[images.length - 1]?.id === selected.id) loadMore();
  }, [selected, images, loadMore]);

  // 菜单"发送到工作台":先取 PNG 元数据,再复用抽屉的同一回调
  async function handleMenuSend(image) {
    try {
      const meta = await api.get(api.albumMetadataUrl(image.id, activeDir));
      onSendToWorkbench?.(meta);
    } catch (err) {
      setError(`读取图片参数失败:${err.message}`);
    }
  }

  // 抽屉"发送到视频生成":图片在服务端本地复制为 I2V 输入图片,再切 tab 预填表单
  async function handleSendToVideo(image) {
    const saved = await api.importVideoInputFromAlbum(image.id, activeDir);
    setSelected(null);
    onSendToVideo?.({ id: saved.id, url: saved.url, name: image.name });
  }

  // 汉堡按钮:开关子目录弹层,打开时拉取当前目录的一级子目录列表
  async function toggleSubPicker() {
    if (subPickerOpen) {
      setSubPickerOpen(false);
      return;
    }
    setSubPickerOpen(true);
    try {
      const data = await api.albumSubdirs(activeDir);
      setSubdirs(data.subdirs);
    } catch (err) {
      setError(`子目录加载失败:${err.message}`);
    }
  }

  async function confirmDelete() {
    if (!pendingDelete || deleting) return;
    setDeleting(true);
    try {
      await api.deleteAlbumImage(pendingDelete.id, activeDir);
      // 前端直接移除该元素,并记录 id 防止在途翻页把它带回
      removedIdsRef.current = new Set([...removedIdsRef.current, pendingDelete.id]);
      setImages((prev) => prev.filter((item) => item.id !== pendingDelete.id));
      if (selected?.id === pendingDelete.id) setSelected(null);
      setPendingDelete(null);
    } catch (err) {
      setError(`删除失败:${err.message}`);
    } finally {
      setDeleting(false);
    }
  }

  // 退出批量选择:清空选择,工具栏随之消失
  function exitBatchMode() {
    setBatchMode(false);
    setBatchSelected(new Set());
  }

  // 批量模式下点击网格项:切换选中状态(不进 lightbox)
  function toggleBatchSelect(image) {
    setBatchSelected((prev) => {
      const next = new Set(prev);
      if (next.has(image.id)) {
        next.delete(image.id);
      } else {
        next.add(image.id);
      }
      return next;
    });
  }

  async function confirmBatchDelete() {
    if (batchDeleting || batchSelected.size === 0) return;
    const targetIds = [...batchSelected];
    setBatchDeleting(true);
    try {
      const result = await api.batchDeleteAlbumImages(targetIds, activeDir);
      // 前端按选中 id 直接移除;仅保留服务端报告删除失败的文件
      const failedIds = new Set((result.failed ?? []).map((f) => f.relpath));
      const removedIds = targetIds.filter((id) => !failedIds.has(id));
      if (removedIds.length) {
        removedIdsRef.current = new Set([...removedIdsRef.current, ...removedIds]);
      }
      setImages((prev) => prev.filter((item) => !removedIds.includes(item.id)));
      if (selected && removedIds.includes(selected.id)) setSelected(null);
      if (result.failed?.length) {
        setError(
          `部分文件删除失败:${result.failed.map((f) => f.relpath).join("、")}`,
        );
      }
      setPendingBatchDelete(false);
      exitBatchMode();
    } catch (err) {
      setError(`批量删除失败:${err.message}`);
    } finally {
      setBatchDeleting(false);
    }
  }

  // 目录 modal 保存:add 创建新 tab 并切入;edit 更新 tab 名称
  async function handleDirSave(name, path) {
    if (dirModal?.mode === "add") {
      const created = await api.addAlbumDir({ name, path });
      setDirs((prev) => [...(prev ?? []), created]);
      setActiveDir(created.id);
    } else if (dirModal?.mode === "edit") {
      const updated = await api.renameAlbumDir(dirModal.dir.id, { name });
      setDirs((prev) => prev.map((d) => (d.id === updated.id ? updated : d)));
    }
    setDirModal(null);
  }

  async function confirmDeleteDir() {
    if (!pendingDeleteDir) return;
    try {
      await api.deleteAlbumDir(pendingDeleteDir.id);
      setDirs((prev) => prev.filter((d) => d.id !== pendingDeleteDir.id));
      if (activeDir === pendingDeleteDir.id) setActiveDir(BUILTIN_DIR);
      setPendingDeleteDir(null);
    } catch (err) {
      setError(`删除目录失败:${err.message}`);
    }
  }

  const selectedIndex = selected
    ? images.findIndex((image) => image.id === selected.id)
    : -1;
  const prevImage = selectedIndex > 0 ? images[selectedIndex - 1] : null;
  const nextImage =
    selectedIndex >= 0 && selectedIndex < images.length - 1
      ? images[selectedIndex + 1]
      : null;
  const activeDirName =
    dirs?.find((d) => d.id === activeDir)?.name ?? activeDir;

  return (
    <>
      <div id="album-tabs" role="tablist" aria-label="相册目录">
        {(dirs ?? [{ id: BUILTIN_DIR, name: "默认", builtin: true }]).map(
          (dir) => (
            <button
              key={dir.id}
              id={`album-tab-${dir.id}`}
              type="button"
              role="tab"
              aria-selected={activeDir === dir.id}
              className={activeDir === dir.id ? "active" : ""}
              title={dir.path}
              onClick={() => {
                setSubDir(null);
                setSubPickerOpen(false);
                setActiveDir(dir.id);
              }}
              onContextMenu={(event) => {
                if (dir.builtin) return;
                event.preventDefault();
                setDirMenu({ x: event.clientX, y: event.clientY, dir });
              }}
            >
              {dir.name}
            </button>
          ),
        )}
        <button
          id="album-tab-add"
          type="button"
          className="album-tab-add"
          aria-label="添加目录"
          title="添加目录"
          onClick={() => setDirModal({ mode: "add" })}
        >
          +
        </button>
      </div>

      <main id="album-page" className={batchMode ? "batch-selecting" : undefined}>
      {error && <div id="album-error" className="form-error">加载失败:{error}</div>}
      {images.length === 0 && !loading && !error && (
        <div className="empty-hint">{subDir ?? activeDirName} 目录中还没有图片</div>
      )}
      <div id="album-grid" className="album-grid">
        {imagesDir === activeDir && imagesSubdir === subDir &&
          images.map((image) => (
            <AlbumItem
              key={image.id}
              image={image}
              dir={activeDir}
              selected={batchMode && batchSelected.has(image.id)}
              onClick={() => {
                // 批量选择状态:点击切换选中,不进 lightbox
                if (batchMode) {
                  toggleBatchSelect(image);
                  return;
                }
                // 图片与视频都在页内弹窗查看(视频用原生播放器)
                setSelected(image);
              }}
              onContextMenu={(event) => {
                event.preventDefault();
                if (batchMode) return; // 批量状态下不弹单个项菜单
                setMenu({ x: event.clientX, y: event.clientY, image });
              }}
            />
          ))}
      </div>
      <div id="album-sentinel" ref={sentinelRef} />
      {loading && <div id="album-loading" className="album-loading">加载中……</div>}
      <button
        id="album-subdir-btn"
        type="button"
        className={subDir ? "active" : undefined}
        aria-label="按子目录浏览"
        title={subDir ? `正在查看子目录:${subDir}` : "按子目录浏览"}
        onClick={toggleSubPicker}
      >
        ☰
      </button>
      {subPickerOpen && (
        <>
          <div
            id="album-subdir-backdrop"
            onClick={() => setSubPickerOpen(false)}
          />
          <div id="album-subdir-panel" role="menu" ref={subPanelRef}>
            <button
              id="album-subdir-all"
              type="button"
              className={subDir === null ? "active" : undefined}
              onClick={() => {
                setSubDir(null);
                setSubPickerOpen(false);
              }}
            >
              返回查看所有相册
            </button>
            {(subdirs ?? []).map((name) => (
              <button
                key={name}
                id={`album-subdir-${name}`}
                type="button"
                className={subDir === name ? "active" : undefined}
                onClick={() => {
                  setSubDir(name);
                  setSubPickerOpen(false);
                }}
              >
                {name}
              </button>
            ))}
            {subdirs !== null && subdirs.length === 0 && (
              <div className="album-subdir-empty">当前目录没有子目录</div>
            )}
          </div>
        </>
      )}
      <button
        id="album-refresh-btn"
        type="button"
        className={loading ? "spinning" : undefined}
        aria-label="刷新相册"
        title="刷新相册"
        disabled={loading}
        onClick={reload}
      >
        ⟳
      </button>

      {batchMode && (
        <div id="album-batch-toolbar" className="album-batch-toolbar">
          <button
            id="album-batch-delete-btn"
            type="button"
            className="danger"
            disabled={batchSelected.size === 0 || batchDeleting}
            onClick={() => setPendingBatchDelete(true)}
          >
            批量删除({batchSelected.size})
          </button>
          <button
            id="album-batch-cancel-btn"
            type="button"
            className="chip"
            onClick={exitBatchMode}
          >
            取消
          </button>
        </div>
      )}

      {menu && (
        <ContextMenu
          x={menu.x}
          y={menu.y}
          onClose={() => setMenu(null)}
          items={[
            {
              id: "album-menu-batch-select",
              label: "批量选择",
              onClick: () => {
                // 右键的该项直接纳入选择,进入批量状态
                setBatchSelected(new Set([menu.image.id]));
                setBatchMode(true);
              },
            },
            // 发送到工作台依赖 PNG 元数据,仅对图片可用
            ...(menu.image.kind === "video"
              ? []
              : [
                  {
                    id: "album-menu-send",
                    label: "发送到工作台",
                    onClick: () => handleMenuSend(menu.image),
                  },
                ]),
            {
              id: "album-menu-delete",
              label: menu.image.kind === "video" ? "删除视频" : "删除图片",
              danger: true,
              onClick: () => setPendingDelete(menu.image),
            },
          ]}
        />
      )}
      {dirMenu && (
        <ContextMenu
          x={dirMenu.x}
          y={dirMenu.y}
          onClose={() => setDirMenu(null)}
          items={[
            {
              id: "album-dir-menu-edit",
              label: "编辑",
              onClick: () => setDirModal({ mode: "edit", dir: dirMenu.dir }),
            },
            {
              id: "album-dir-menu-delete",
              label: "删除",
              danger: true,
              onClick: () => setPendingDeleteDir(dirMenu.dir),
            },
          ]}
        />
      )}

      {dirModal && (
        <DirFormModal
          mode={dirModal.mode}
          dir={dirModal.dir}
          onSave={handleDirSave}
          onClose={() => setDirModal(null)}
        />
      )}
      {pendingDeleteDir && (
        <Modal
          id="album-dir-delete-modal"
          title="删除目录"
          onClose={() => setPendingDeleteDir(null)}
          footer={
            <>
              <button
                id="album-dir-delete-cancel"
                type="button"
                className="chip"
                onClick={() => setPendingDeleteDir(null)}
              >
                取消
              </button>
              <button
                id="album-dir-delete-confirm"
                type="button"
                className="danger"
                onClick={confirmDeleteDir}
              >
                确认删除
              </button>
            </>
          }
        >
          <p className="modal-text">
            将把目录「{pendingDeleteDir.name}」从相册移除(不会删除磁盘上的文件),确认删除?
          </p>
        </Modal>
      )}
      {pendingBatchDelete && (
        <Modal
          id="album-batch-delete-modal"
          title="批量删除"
          onClose={() => setPendingBatchDelete(false)}
          closeDisabled={batchDeleting}
          footer={
            <>
              <button
                id="album-batch-delete-cancel"
                type="button"
                className="chip"
                disabled={batchDeleting}
                onClick={() => setPendingBatchDelete(false)}
              >
                取消
              </button>
              <button
                id="album-batch-delete-confirm"
                type="button"
                className="danger"
                disabled={batchDeleting}
                onClick={confirmBatchDelete}
              >
                {batchDeleting ? "删除中……" : "确认删除"}
              </button>
            </>
          }
        >
          <p className="modal-text">
            将从本机删除选中的 {batchSelected.size} 个图片/视频,确认删除?
          </p>
        </Modal>
      )}
      {pendingDelete && (
        <Modal
          id="album-delete-modal"
          title={pendingDelete.kind === "video" ? "删除视频" : "删除图片"}
          onClose={() => setPendingDelete(null)}
          closeDisabled={deleting}
          footer={
            <>
              <button
                id="album-delete-cancel"
                type="button"
                className="chip"
                disabled={deleting}
                onClick={() => setPendingDelete(null)}
              >
                取消
              </button>
              <button
                id="album-delete-confirm"
                type="button"
                className="danger"
                disabled={deleting}
                onClick={confirmDelete}
              >
                {deleting ? "删除中……" : "确认删除"}
              </button>
            </>
          }
        >
          <p className="modal-text">
            将从本机删除该{pendingDelete.kind === "video" ? "视频" : "图片"}({pendingDelete.name}),确认删除?
          </p>
        </Modal>
      )}

      {selected && (
        <Lightbox
          imageUrl={api.albumImageUrl(selected.id, activeDir)}
          metadataUrl={api.albumMetadataUrl(selected.id, activeDir)}
          fallback={{ width: selected.width, height: selected.height }}
          fileInfo={{
            location: selected.id,
            sizeBytes: selected.size_bytes,
            mtimeNs: selected.mtime_ns,
          }}
          kind={selected.kind ?? "image"}
          onClose={() => setSelected(null)}
          onPrev={prevImage ? () => setSelected(prevImage) : null}
          onNext={nextImage ? () => setSelected(nextImage) : null}
          onSendToWorkbench={
            selected.kind === "video"
              ? // 视频:把生成参数预填进视频生成表单
                onSendVideoMeta
                  ? (meta) => {
                      setSelected(null);
                      onSendVideoMeta(meta);
                    }
                  : null
              : onSendToWorkbench
                ? (meta) => {
                    setSelected(null);
                    onSendToWorkbench(meta);
                  }
                : null
          }
          onSendToVideo={
            // 「发送到视频生成」仅对图片(作为 I2V 输入图)开放
            selected.kind !== "video" && onSendToVideo
              ? () => handleSendToVideo(selected)
              : null
          }
        />
      )}
      </main>
    </>
  );
}

// 目录新增/编辑弹窗:新增填名称 + 位置;编辑仅可改名称
function DirFormModal({ mode, dir, onSave, onClose }) {
  const [name, setName] = useState(dir?.name ?? "");
  const [path, setPath] = useState("");
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    if (saving) return;
    if (!name.trim()) {
      setError("目录名称不能为空");
      return;
    }
    if (mode === "add" && !path.trim()) {
      setError("目录位置不能为空");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      await onSave(name.trim(), path.trim());
    } catch (err) {
      setError(err.message);
      setSaving(false);
    }
  }

  return (
    <Modal
      id="album-dir-modal"
      title={mode === "add" ? "添加目录" : "编辑目录"}
      onClose={onClose}
      closeDisabled={saving}
      footer={
        <>
          <button
            id="album-dir-cancel"
            type="button"
            className="chip"
            disabled={saving}
            onClick={onClose}
          >
            取消
          </button>
          <button
            id="album-dir-save"
            type="submit"
            form="album-dir-form"
            className="primary"
            disabled={saving}
          >
            {saving ? "保存中……" : "保存"}
          </button>
        </>
      }
    >
      <form id="album-dir-form" className="form" onSubmit={handleSubmit}>
        <div className="field">
          <label htmlFor="album-dir-name">目录名称</label>
          <input
            id="album-dir-name"
            type="text"
            value={name}
            placeholder="展示在 tab 上的名称"
            autoFocus
            onChange={(e) => setName(e.target.value)}
          />
        </div>
        {mode === "add" && (
          <div className="field">
            <label htmlFor="album-dir-path">目录位置</label>
            <input
              id="album-dir-path"
              type="text"
              value={path}
              placeholder="本机目录的绝对路径,如 E:/images"
              onChange={(e) => setPath(e.target.value)}
            />
          </div>
        )}
        {error && <div id="album-dir-error" className="form-error">{error}</div>}
      </form>
    </Modal>
  );
}

// 单个相册项:进入视口附近才挂载 <img>/<video>,移出后卸载以回收资源;
// 占位用 aspect-ratio 保持网格布局稳定;加载失败(如文件刚被删)整格隐藏。
function AlbumItem({ image, dir, selected, onClick, onContextMenu }) {
  const ref = useRef(null);
  const [active, setActive] = useState(false);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => setActive(entry.isIntersecting),
      { rootMargin: VIEWPORT_MARGIN },
    );
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);

  if (failed) return null;

  return (
    <div
      ref={ref}
      id={`album-item-${image.id}`}
      className={selected ? "album-item selected" : "album-item"}
      onClick={onClick}
      onContextMenu={onContextMenu}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && onClick()}
    >
      {active &&
        (image.kind === "video" ? (
          // 视频封面用服务端抽取的首帧静态图(iOS 不会渲染 <video> 的首帧)
          <img
            src={api.albumPosterUrl(image.id, dir)}
            alt={image.name}
            loading="lazy"
            onError={() => setFailed(true)}
          />
        ) : (
          <img
            src={api.albumImageUrl(image.id, dir)}
            alt={image.name}
            loading="lazy"
            onError={() => setFailed(true)}
          />
        ))}
      {image.kind === "video" && <span className="album-kind-badge">视频</span>}
      {selected && (
        <span className="album-item-check" aria-label="已选择">
          ✓
        </span>
      )}
    </div>
  );
}
