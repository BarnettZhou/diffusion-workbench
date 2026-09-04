// 3:4 透明占位图:给「暂无封面 / 隐私模式」占位提供文档流内的高度来源。
// 用替换元素(img)的内在比例撑高,规避 WebKit 在 grid/flex 内容尺寸计算阶段
// 把非替换元素的百分比 padding / aspect-ratio 高度算成 0 的缺陷(见 styles.css
// .model-cover 注释)。文字由占位 div 里的 <span> 绝对定位叠加。
export const COVER_SPACER =
  "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='3' height='4'/%3E";
