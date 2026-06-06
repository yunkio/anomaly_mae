/* Bind react-plotly.js to the prebuilt plotly.js-dist-min bundle (avoids compiling
 * the full plotly.js source — smaller, more reliable build). This is the single Plot
 * component the whole app imports. */
import createPlotlyComponent from "react-plotly.js/factory";
// @ts-ignore - dist-min has no types; declared in vite-env.d.ts
import Plotly from "plotly.js-dist-min";

const Plot = createPlotlyComponent(Plotly);
export default Plot;
