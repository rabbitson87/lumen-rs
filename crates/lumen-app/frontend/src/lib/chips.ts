/** One choice of a segmented control of preset chips. */
export type Chip = {
  value: string;
  label: string;
  /** A second line under the label (an estimate, a state). */
  sub?: string;
  /** The tooltip; the reason when the chip is disabled. */
  title?: string;
  disabled?: boolean;
  /** Drawn as a warning: the choice is allowed but has a catch. */
  warn?: boolean;
};
