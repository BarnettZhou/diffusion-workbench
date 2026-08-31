export default function TextEncoderSelector({ idPrefix, localEncoders, localIndex, onLocalIndexChange, remoteIds = [], source, onSourceChange, remoteId, onRemoteIdChange }) {
  return (
    <div id={`${idPrefix || "main"}-text-encoder-card`} className="size-card-embedded form text-encoder-card">
      <div className="card-header">
        <h2>文本编码器</h2>
        <div id={`${idPrefix || "main"}-text-encoder-source`} className="size-mode-switch" role="tablist" aria-label="文本编码器来源">
          <button type="button" role="tab" aria-selected={source === "local"} className={source === "local" ? "active" : ""} onClick={() => onSourceChange("local")}>本地</button>
          <button type="button" role="tab" aria-selected={source === "remote"} className={source === "remote" ? "active" : ""} disabled={!remoteIds.length} onClick={() => onSourceChange("remote")}>远端</button>
        </div>
      </div>
      {source === "remote" ? (
        <div className="field"><label htmlFor={`${idPrefix || "main"}-remote-text-encoder-id`}>远端 Text Encoder ID</label><select id={`${idPrefix || "main"}-remote-text-encoder-id`} value={remoteId ?? ""} disabled={!remoteIds.length} onChange={(e) => onRemoteIdChange(e.target.value)}>{remoteIds.map((id) => <option key={id} value={id}>{id}</option>)}</select></div>
      ) : (
        <div className="field"><label htmlFor={`${idPrefix || "main"}-local-text-encoder-select`}>本地 Text Encoder</label><select id={`${idPrefix || "main"}-local-text-encoder-select`} value={localIndex ?? ""} disabled={!localEncoders.length} onChange={(e) => onLocalIndexChange(Number(e.target.value))}>{localEncoders.map((item) => <option key={item.index} value={item.index}>{item.display_name}</option>)}</select></div>
      )}
    </div>
  );
}
