// Narra frontend runtime config — picks the backend automatically by host:
//
//   • localhost / 127.0.0.1  → "" (same origin): local dev hits your local backend.
//   • *.hf.space             → "" (same origin): the HF Space serves this page itself.
//   • anything else (Netlify)→ the HF Space backend URL.
//
// So local dev, the Space's own UI, and the Netlify deploy all work from one file.
// If your Space URL changes, update the one string below.
(function () {
  var h = location.hostname;
  var sameOrigin = h === "localhost" || h === "127.0.0.1" || h === "" || h.endsWith(".hf.space");
  window.NARRA_API_BASE = sameOrigin ? "" : "https://zitani47-narra-tts.hf.space";
})();
