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
export default function AlbumPage({ onSendToWorkbench }) {
  const [dirs, setDirs] = useState(null);
  const [activeDir, setActiveDir] = useState(BUILTIN_DIR);
  const [images, setImages] = useState([]);
  // 当前 images 列表所属的目录;切目录后的旧列表不再渲染,避免旧 relpath 配新 dir 请求 404
  const [imagesDir, setImagesDir] = useState(BUILTIN_DIR);
  const [cursor, setCursor] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);
  // 图片右键菜单:{x, y, image};待确认删除的图片;删除中状态
  const [menu, setMenu] = useState(null);
  const [pendingDelete, setPendingDelete] = useState(null);
  const [deleting, setDeleting] = useState(false);
  // 目录 tabs:add/edit modal 开关与编辑对象;tab 右键菜单;待确认删除的目录
  const [dirModal, setDirModal] = useState(null); // {mode: "add" | "edit", dir?}
  const [dirMenu, setDirMenu] = useState(null); // {x, y, dir}
  const [pendingDeleteDir, setPendingDeleteDir] = useState(null);
  const sentinelRef = useRef(null);
  const stateRef = useRef({ cursor: null, loading: false, dir: BUILTIN_DIR });
  stateRef.current = { cursor, loading, dir: activeDir };

  const loadMore = useCallback(async () => {
    const { cursor: current, loading: busy, dir } = stateRef.current;
    if (busy || current === "end") return;
    setLoading(true);
    try {
      const page = await api.album(PAGE_SIZE, current, dir);
      // 请求发出后目录已切换:丢弃过期页,loading 交给新目录的请求收尾
      if (stateRef.current.dir !== dir) return;
      setImages((prev) => [...prev, ...page.images]);
      setImagesDir(dir);
      setCursor(page.next_cursor ?? "end");
    } catch (err) {
      if (stateRef.current.dir === dir) setError(err.message);
    } finally {
      if (stateRef.current.dir === dir) setLoading(false);
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
    stateRef.current = { cursor: null, loading: false, dir: activeDir };
    loadMore();
  }, [activeDir, loadMore]);

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

  async function confirmDelete() {
    if (!pendingDelete || deleting) return;
    setDeleting(true);
    try {
      await api.deleteAlbumImage(pendingDelete.id, activeDir);
      setImages((prev) => prev.filter((item) => item.id !== pendingDelete.id));
      if (selected?.id === pendingDelete.id) setSelected(null);
      setPendingDelete(null);
    } catch (err) {
      setError(`删除失败:${err.message}`);
    } finally {
      setDeleting(false);
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
    <main id="album-page">
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
              onClick={() => setActiveDir(dir.id)}
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

      {error && <div id="album-error" className="form-error">加载失败:{error}</div>}
      {images.length === 0 && !loading && !error && (
        <div className="empty-hint">{activeDirName} 目录中还没有图片</div>
      )}
      <div id="album-grid" className="album-grid">
        {imagesDir === activeDir &&
          images.map((image) => (
            <AlbumItem
              key={image.id}
              image={image}
              dir={activeDir}
              onClick={() => setSelected(image)}
              onContextMenu={(event) => {
                event.preventDefault();
                setMenu({ x: event.clientX, y: event.clientY, image });
              }}
            />
          ))}
      </div>
      <div id="album-sentinel" ref={sentinelRef} />
      {loading && <div id="album-loading" className="album-loading">加载中……</div>}
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

      {menu && (
        <ContextMenu
          x={menu.x}
          y={menu.y}
          onClose={() => setMenu(null)}
          items={[
            {
              id: "album-menu-send",
              label: "发送到工作台",
              onClick: () => handleMenuSend(menu.image),
            },
            {
              id: "album-menu-delete",
              label: "删除图片",
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
      {pendingDelete && (
        <Modal
          id="album-delete-modal"
          title="删除图片"
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
            将从本机删除该图片({pendingDelete.name}),确认删除?
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
          onClose={() => setSelected(null)}
          onPrev={prevImage ? () => setSelected(prevImage) : null}
          onNext={nextImage ? () => setSelected(nextImage) : null}
          onSendToWorkbench={
            onSendToWorkbench
              ? (meta) => {
                  setSelected(null);
                  onSendToWorkbench(meta);
                }
              : null
          }
        />
      )}
    </main>
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

// 单个图片项:进入视口附近才挂载 <img>,移出后卸载以回收资源;
// 占位用 aspect-ratio 保持网格布局稳定;加载失败(如文件刚被删)整格隐藏。
function AlbumItem({ image, dir, onClick, onContextMenu }) {
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
      className="album-item"
      onClick={onClick}
      onContextMenu={onContextMenu}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && onClick()}
    >
      {active && (
        <img
          src={api.albumImageUrl(image.id, dir)}
          alt={image.name}
          loading="lazy"
          onError={() => setFailed(true)}
        />
      )}
    </div>
  );
}
