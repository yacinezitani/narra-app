// Narra frontend runtime config.
//
// This is the ONE file you edit per deployment — no build step needed.
//
//   • Local dev (FastAPI serves this page):  leave it empty ("") = same origin.
//   • Static deploy (Vercel / Netlify):      set it to your backend's URL, i.e.
//     the Hugging Face Space, e.g. "https://yourname-narra-tts.hf.space".
//
// The backend must allow this site's origin via its ALLOWED_ORIGINS env var.
window.NARRA_API_BASE = "";
