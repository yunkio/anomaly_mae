/* FB-8 (frontend-only, per RV2-04) — config hide-inactive.
 *
 * When a feature flag is OFF its dependent keys are meaningless. This computes the set
 * of INACTIVE keys client-side from the flat /config dict + a flag → dependent-key-group
 * map grounded in mae_anomaly/config.py (verified line-for-line in change-spec FB-8(b)):
 *   use_grl=false           → grl_* + wdgrl_*
 *   use_feature_matching=false → fm_*
 *   use_scad=false          → scad_*
 *   margin_type != 'dynamic'→ dynamic_margin_k
 *   use_teacher_output_ema=false → teacher_output_ema_momentum
 *   anomaly_score_mode != 'adaptive' → score_recon_disc_ratio, eval_disc_weight, eval_fm_weight
 *
 * The toggle reveals inactive keys (nothing is irreversibly hidden — the payload keeps
 * them). The map is ADDITIVE, not a whitelist: an unknown flag/key is never hidden (F5);
 * when the deciding flag is ABSENT, the group is treated as ACTIVE (graceful). */

type FlagRule =
  | { flag: string; offWhen: (v: any) => boolean; prefixes?: string[]; keys?: string[] };

const RULES: FlagRule[] = [
  { flag: "use_grl", offWhen: (v) => v === false, prefixes: ["grl_", "wdgrl_"] },
  { flag: "use_feature_matching", offWhen: (v) => v === false, prefixes: ["fm_"] },
  { flag: "use_scad", offWhen: (v) => v === false, prefixes: ["scad_"] },
  { flag: "margin_type", offWhen: (v) => v !== "dynamic", keys: ["dynamic_margin_k"] },
  {
    flag: "use_teacher_output_ema",
    offWhen: (v) => v === false,
    keys: ["teacher_output_ema_momentum"],
  },
  {
    flag: "anomaly_score_mode",
    offWhen: (v) => v !== "adaptive",
    keys: ["score_recon_disc_ratio", "eval_disc_weight", "eval_fm_weight"],
  },
];

export interface ConfigActivity {
  /** keys that are inactive (their deciding flag is OFF). Never includes the flag key. */
  inactive: Set<string>;
  /** human note per inactive key — why it is hidden. */
  reason: Record<string, string>;
}

/** Compute the inactive-key set for one leaf's flat config. */
export function computeConfigActivity(config: Record<string, any> | undefined | null): ConfigActivity {
  const inactive = new Set<string>();
  const reason: Record<string, string> = {};
  const cfg = config ?? {};
  for (const rule of RULES) {
    // graceful: if the deciding flag is ABSENT, treat the group as ACTIVE (don't hide).
    if (!(rule.flag in cfg)) continue;
    const off = rule.offWhen(cfg[rule.flag]);
    if (!off) continue;
    const why = `${rule.flag} = ${JSON.stringify(cfg[rule.flag])} (inactive)`;
    for (const k of Object.keys(cfg)) {
      if (k === rule.flag) continue;
      const hitPrefix = rule.prefixes?.some((p) => k.startsWith(p));
      const hitKey = rule.keys?.includes(k);
      if (hitPrefix || hitKey) {
        inactive.add(k);
        reason[k] = why;
      }
    }
  }
  return { inactive, reason };
}
