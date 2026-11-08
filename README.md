# Promo Finder (public demo)

Streamlit app over precomputed promo rankings for microdrama shows.
Set the environment variable / Streamlit secret `PUBLIC_MODE=1`: full screenplay excerpts are not included in this repo
and are hidden; timestamps, hook and cliffhanger lines, scores, rankings, explanations, evaluation, failure analysis and
feedback-learning status are shown. Reviews made here go to a separate demo file and never change the ranking.
Run locally: `pip install -r requirements.txt` then `PUBLIC_MODE=1 streamlit run app.py`.
