# Promo Finder (public demo)

Streamlit app over precomputed promo rankings for microdrama shows.
Set the environment variable / Streamlit secret `PUBLIC_MODE=1`: full screenplay excerpts are not included in this repo
and are hidden; timestamps, hook and cliffhanger lines, scores, rankings, explanations, evaluation, failure analysis and
feedback-learning status are shown. Reviews made here go to a separate demo file and never change the ranking.
Run locally: `pip install -r requirements.txt` then `PUBLIC_MODE=1 streamlit run app.py`.

**Analyze New Input** runs the real pipeline on uploaded `.txt` screenplays or a `.zip` of them, in a temporary workspace
outside this repository. It needs the server-side environment variable `GEMINI_API_KEY` (optionally `GEMINI_MODEL`,
`LLM_MIN_INTERVAL`); without it the pre-computed demo still works and the live page explains what is missing.
