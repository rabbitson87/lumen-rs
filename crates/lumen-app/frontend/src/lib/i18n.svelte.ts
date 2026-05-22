/**
 * Tiny i18n runtime.
 *
 * Why home-grown instead of `svelte-i18n` / `paraglide`:
 *   - the dictionary is a single flat record, no plural/format ICU rules
 *     are required yet, and the app only ships two locales today.
 *   - keeps zero runtime deps and avoids a bundler regression risk during
 *     the Vite + Tauri pre-build step.
 *
 * Reactive model: `locale` is a Svelte 5 `$state` rune. The exported `t()`
 * is a *plain function* that reads `locale.value` — Svelte's fine-grained
 * reactivity tracks the read inside any template `{t("...")}` expression,
 * so a locale switch re-renders every translated string without manual
 * subscription bookkeeping.
 *
 * Persistence: the chosen locale is mirrored to `localStorage` so
 * subsequent loads (including app restarts) restore the user's choice.
 */
import { en } from "../messages/en";
import { ko } from "../messages/ko";

export type Locale = "en" | "ko";

const LOCALE_KEY = "lumen.locale";
const SUPPORTED: readonly Locale[] = ["en", "ko"] as const;

function loadInitial(): Locale {
  if (typeof localStorage === "undefined") return "en";
  const v = localStorage.getItem(LOCALE_KEY);
  return (SUPPORTED as readonly string[]).includes(v ?? "") ? (v as Locale) : "en";
}

type Dict = Record<string, string>;
const DICTS: Record<Locale, Dict> = { en, ko };

class I18n {
  locale = $state<Locale>(loadInitial());

  set(loc: Locale) {
    this.locale = loc;
    if (typeof localStorage !== "undefined") {
      localStorage.setItem(LOCALE_KEY, loc);
    }
  }
}

export const i18n = new I18n();

/**
 * Lookup a translation key. Falls back to the English dictionary, then to
 * the raw key — so missing keys are visible during development without
 * crashing the render.
 */
export function t(key: string): string {
  const dict = DICTS[i18n.locale];
  return dict[key] ?? en[key] ?? key;
}

/**
 * Convenience metadata for the language picker UI.
 */
export const LOCALE_LABELS: Record<Locale, string> = {
  en: "English",
  ko: "한국어",
};
