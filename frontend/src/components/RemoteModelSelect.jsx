// 远端(LLM / 反推 API)端点 × 模型的统一下拉选项。
// 把 settings 里的 endpoints 与当前 selected 序列化成 `${endpoint.id}::${model}` 的 option value,
// 切换后由调用方把回调解析回 {endpoint_id, model} 写回 settings.selected。
//
// selected 缺失或引用的端点/模型已不存在时,显示占位「未选择模型」,value 为 ""。
// endpoints 为空时整个下拉禁用并提示先添加端点。

function makeValue(endpointId, model) {
  return `${endpointId}::${model}`;
}

function parseValue(value) {
  if (!value || typeof value !== "string") return { endpoint_id: "", model: "" };
  const sep = value.indexOf("::");
  if (sep === -1) return { endpoint_id: "", model: "" };
  const endpoint_id = value.slice(0, sep);
  const model = value.slice(sep + 2);
  if (!endpoint_id || !model) return { endpoint_id: "", model: "" };
  return { endpoint_id, model };
}

function endpointLabel(endpoint) {
  return endpoint?.name?.trim() || endpoint?.base_url?.trim() || "未命名端点";
}

export default function RemoteModelSelect({
  endpoints,
  selected,
  onChange,
  idPrefix = "remote-model-select",
  disabled = false,
}) {
  const list = Array.isArray(endpoints) ? endpoints : [];
  const endpointId = selected?.endpoint_id ?? "";
  const model = selected?.model ?? "";
  // 渲染当前选中的 option:即便端点或模型已不存在,也临时插入一条以便 select 显示。
  const validValue = list.some(
    (endpoint) => endpoint.id === endpointId && Array.isArray(endpoint.models) && endpoint.models.includes(model),
  );
  const currentValue = validValue ? makeValue(endpointId, model) : "";

  function handleChange(event) {
    const parsed = parseValue(event.target.value);
    onChange?.(parsed);
  }

  if (list.length === 0) {
    return (
      <select id={idPrefix} className="remote-model-select" disabled>
        <option value="">请先在设置中添加端点</option>
      </select>
    );
  }

  return (
    <select
      id={idPrefix}
      className="remote-model-select"
      value={currentValue}
      disabled={disabled}
      onChange={handleChange}
    >
      <option value="">未选择模型</option>
      {list.map((endpoint) => {
        const models = Array.isArray(endpoint.models) ? endpoint.models : [];
        if (models.length === 0) return null;
        const groupLabel = endpointLabel(endpoint);
        return (
          <optgroup key={endpoint.id} label={groupLabel}>
            {models.map((name) => (
              <option key={`${endpoint.id}::${name}`} value={makeValue(endpoint.id, name)}>
                {name}
              </option>
            ))}
          </optgroup>
        );
      })}
    </select>
  );
}