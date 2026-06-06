import { Routes, Route } from "react-router-dom";
import AppShell from "./components/AppShell";
import Overview from "./views/Overview";
import ExperimentDetail from "./views/ExperimentDetail";
import DatasetDetail from "./views/DatasetDetail";
import Rankings from "./views/Rankings";
import CompareWorkbench from "./views/CompareWorkbench";
import Explorer from "./views/Explorer";
import ConfigView from "./views/ConfigView";
import Gallery from "./views/Gallery";
import GifStudio from "./views/GifStudio";

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/exp/:expId" element={<ExperimentDetail />} />
        <Route path="/exp/:expId/ds/:dsKey" element={<DatasetDetail />} />
        <Route path="/rankings" element={<Rankings />} />
        <Route path="/compare" element={<CompareWorkbench />} />
        <Route path="/explorer" element={<Explorer />} />
        <Route path="/config" element={<ConfigView />} />
        <Route path="/gallery" element={<Gallery />} />
        <Route path="/gif" element={<GifStudio />} />
        <Route path="*" element={<Overview />} />
      </Routes>
    </AppShell>
  );
}
