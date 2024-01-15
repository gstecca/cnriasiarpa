aggiorna ubuntu:
	sudo apt-get update
	sudo apt-get upgrade

aggiorna pip:
	python -m pip install --upgrade pip

Aggiorna Conda:
	conda update conda
	conda update --all 

Crea un nuovo environment, chiamato ad es. ARPA_venv, con python=3.11
	conda create -n ARPA_venv python=3.11

aggiorna build-essential:
	sudo apt install build-essential

installa i pacchetti dal canale conda-forge, volendo non nel venv ma da fuori, ad esempio dal base, con 
	conda install -c conda-forge -n ARPA_venv nome_pacchetto

Quindi installa streamlit, folium e streamlit-folium

	conda install -c conda-forge -n ARPA_venv streamlit
	conda install -c conda-forge -n ARPA_venv folium
	conda install -c conda-forge -n ARPA_venv streamlit-folium

da terminale, una volta attivato il venv, installare i pacchetti del file requiremnts.txt













