# cnriasiarpa
copyright Istituto di Analisi dei Sistemi ed Informatica "Antonio Ruberti" del Consiglio Nazionale delle Ricerche, via dei Taurini 19, 00185.
Authors: Marco Boresta and Diego Maria Pinto and Giuseppe Stecca
For any inquirey please contact direzione@iasi.cnr.it, giuseppe.stecca@iasi.cnr.it, diegomaria.pinto@iasi.cnr.it, marco.boresta@iasi.cnr.it

- usage of dashboard agv without docker:\
  prerequisites: create a conda environment with all requirements and name it arpa \
  open a terminal window \
  `$ conda activate arpa` \
  `$ cd cnriasiarpa/arpaagv` \
  `$ streamlit run agv_analysis_UI.py --server.port 8501` 

  once server started, open browser page listed in the terminal log (eg. http://127.0.0.1:8501/   or  http://localhost:8501/) \
  If you do not know the raspberry IP and you want to visit the app from outside use the following command  `$ hostname -I` \
  to close the app press CTRL+C on the terminal 1
  \

- usage of dashboard agv with docker:\
  prerequisite: docker, docker-compose \
  to build the container navigate on the arpaagv folder and run: \
  `$ docker compose up --build` \
  visit the url http://localhost:8501 to visulize the app \
  to stop the container press CTRL+C \
  to run the app without rebuild (if there are no changes to the git) digit the following command: \
   `$ docker compose up` 


- usage of dashboard logistics: \
  prerequisites: create a conda environment with all requirements and name it arpa \
  open a terminal window \
  `$ conda activate arpa ` \
  `$ cd cnriasiarpa/arpascm ` \
  `$ streamlit run ARPA_logistics_app.py --server.port 8502` 

  once server started, open browser page listed in the terminal log (eg. http://127.0.0.1:8502/   or  http://localhost:8502/) \
  If you do not know the raspberry IP and you want to visit the app from outside use the following command  `$ hostname -I` \
  to close the app press CTRL+C on the terminal 
