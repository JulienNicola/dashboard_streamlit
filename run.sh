apt update
apt-get install -y libgomp1
python -m streamlit run streamlit_app.py --server.port 8000 --server.address 0.0.0.0