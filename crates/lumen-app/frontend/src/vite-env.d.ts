/// <reference types="svelte" />
/// <reference types="vite/client" />

// Side-effect CSS imports (`import "./app.css"`) have no type on their own, so
// `svelte-check` reported "Cannot find module or type declarations" on every
// run. A permanent error trains people to skim the checker output, which is
// how the `errored` dead comparison beside it survived.
declare module "*.css";
