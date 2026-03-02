README – How to Run the Project (Python 3.12)
	1.	Install Python 3.12

Download from:
https://www.python.org/downloads/

After installing, verify in terminal:

python3.12 –version

You should see:
Python 3.12.x
	2.	Create a Virtual Environment

From the project folder, run:

python3.12 -m venv venv

Activate it:

Mac/Linux:
source venv/bin/activate

Windows:
venv\Scripts\activate

Check version:

python -V

It should say:
Python 3.12.x
	3.	Install Dependencies

If there is a requirements.txt file:

pip install -r requirements.txt

If not, install manually:

pip install streamlit scikit-learn xgboost lightgbm pandas numpy joblib
	4.	Train the Models

Run:

python train.py

Wait for it to finish. This will generate the saved model files.
	5.	Run the Streamlit App

Run:

streamlit run app.py

It should open automatically in your browser.

If not, click the Local URL shown in the terminal.

Next Time You Open the Project

Just activate the environment and run Streamlit:

source venv/bin/activate
streamlit run app.py
