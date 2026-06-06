/* The 4 canonical async states (AC-F3.1), from one shared wrapper so they are
 * uniform + testable: loading skeleton / labeled empty / partial ribbon / error chip.
 * Bound to a TanStack-Query-style result + the backend's present/{available}/errors. */
import type { ReactNode } from "react";

interface Props<T> {
  query: {
    isLoading: boolean;
    isError: boolean;
    error?: unknown;
    data?: T;
  };
  /** the panel body, given non-null data */
  children: (data: T) => ReactNode;
  /** detect an "empty" / "not available for this run" payload from the backend */
  isEmpty?: (data: T) => boolean;
  emptyLabel?: string;
  /** detect a partial/in-progress payload (renders body + a running ribbon) */
  isPartial?: (data: T) => boolean;
  partialLabel?: string;
  /** detect a backend-reported soft error (errors[] / {error}) on a 200 */
  softError?: (data: T) => string | null;
  height?: number;
  title?: string;
}

export default function AsyncPanel<T>({
  query,
  children,
  isEmpty,
  emptyLabel = "No data for this selection.",
  isPartial,
  partialLabel = "Still running — showing data up to the last written epoch.",
  softError,
  height = 240,
  title,
}: Props<T>) {
  if (query.isLoading) {
    return (
      <div>
        {title && <div className="card-title">{title}</div>}
        <div className="async-skel" style={{ height }} aria-busy="true" aria-label="loading" />
      </div>
    );
  }
  if (query.isError) {
    return (
      <div>
        {title && <div className="card-title">{title}</div>}
        <div className="async-state error" style={{ minHeight: Math.min(height, 160) }}>
          <span className="glyph">⚠</span>
          <div>Request failed</div>
          <div className="mono subtle">{String((query.error as Error)?.message ?? query.error)}</div>
        </div>
      </div>
    );
  }
  const data = query.data;
  if (data === undefined || data === null) {
    return (
      <div>
        {title && <div className="card-title">{title}</div>}
        <div className="async-state">
          <span className="glyph">∅</span>
          <div>{emptyLabel}</div>
        </div>
      </div>
    );
  }
  const soft = softError?.(data);
  if (soft) {
    return (
      <div>
        {title && <div className="card-title">{title}</div>}
        <div className="async-state error" style={{ minHeight: Math.min(height, 140) }}>
          <span className="glyph">⚠</span>
          <div>{soft}</div>
        </div>
      </div>
    );
  }
  if (isEmpty?.(data)) {
    return (
      <div>
        {title && <div className="card-title">{title}</div>}
        <div className="async-state">
          <span className="glyph">∅</span>
          <div>{emptyLabel}</div>
        </div>
      </div>
    );
  }
  const partial = isPartial?.(data);
  return (
    <div>
      {title && (
        <div className="card-title">
          {title}
          {partial && <span className="ribbon">● {partialLabel}</span>}
        </div>
      )}
      {!title && partial && (
        <div style={{ marginBottom: 8 }}>
          <span className="ribbon">● {partialLabel}</span>
        </div>
      )}
      {children(data)}
    </div>
  );
}
