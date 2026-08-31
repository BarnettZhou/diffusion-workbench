export default function TextEncoderSelector({ idPrefix, localEncoders, localIndex, onLocalIndexChange, remoteIds = [], source, onSourceChange, remoteId, onRemoteIdChange }) {
  return (
    <div className="field" id={`${idPrefix}-text-encoder`}>
      <label htmlFor={`${idPrefix}-text-encoder-source`}>文本编码器 Text Encoder</label>
      <select id={`${idPrefix}-text-encoder-source`} value={source} onChange={(e) => onSourceChange(e.target.value)}>
        <option value="local">本地</option>
        <option value="remote" disabled={!remoteIds.length}>远端</option>
      </select>
      {source === "remote" ? (
        <select id={`${idPrefix}-remote-text-encoder-id`} value={remoteId ?? ""} disabled={!remoteIds.length} onChange={(e) => onRemoteIdChange(e.target.value)}>
          {remoteIds.map((id) => <option key={id} value={id}>{id}</option>)}
        </select>
      ) : (
        <select id={`${idPrefix}-local-text-encoder-select`} value={localIndex ?? ""} disabled={!localEncoders.length} onChange={(e) => onLocalIndexChange(Number(e.target.value))}>
          {localEncoders.map((item) => <option key={item.index} value={item.index}>{item.display_name}</option>)}
        </select>
      )}
    </div>
  );
}
