python3 -m venv venv
source venv/bin/activate
pip install numpy
pip install torch
pip freeze > requirement.txt
python cosine_similarity.py
