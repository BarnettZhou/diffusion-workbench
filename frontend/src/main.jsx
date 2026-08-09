import { createRoot } from "react-dom/client";
import App from "./App";
import { MessageProvider } from "./components/Message";
import "./styles.css";

createRoot(document.getElementById("root")).render(
  <MessageProvider>
    <App />
  </MessageProvider>,
);
