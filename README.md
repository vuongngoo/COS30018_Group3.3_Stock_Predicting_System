
***

# Stock Predicting System
Project Option C - COS30018 - Intelligent System - Swinburne University

This is the solo project of Group 3.3 (Ngo Sy Vuong).

**About the Github repository:**

The Code section contains the source code of the Project, the Requirements of each project phase (7 phases) and the PDF files of the Reports.

For more information about the Project's Reports and Environment Setup, please see the Wiki of this repository.

***

### 1. Environment
The system is built on an isolated virtual environment (venv) to ensure dependency abstraction and eliminate version conflicts with global system libraries:
* Core Runtime: **Python 3.11.x (64-bit)**
* Execution Interface: **Windows Command Prompt (CMD)**

***

### 2. Dependencies
* **tensorflow**: Core Deep Learning framework used to compile structure and execute the LSTM network
* **yfinance**: Automated financial data retrieval tool used to fetch historical structural data for CBA.AX via the Yahoo Finance API
* **scikit-learn**: Provides data preprocessing mechanics
(specifically utilized for the MinMaxScaler engine to normalize array bounds between 0 and 1)
* **pandas**: Sequential series concatenation, index cleaning and CSV file extraction
* **numpy**: Low-level vectorized array manipulation, matrix dimension reshaping and numeric tensor optimization
* **matplotlib**: Render visual evaluations

***

### 3. Deployment
To reconstruct this environment layout from scratch, run the following sequential commands within the target file directory: **(make sure you have installed Python 3.11)**

Initialize the Python 3.11 Isolation layer

`py -3.11 -m venv venv`

Activate the localized scripts block (Windows)

`venv\Scripts\activate.bat`

Synchronize deployment prerequisites

`pip install -r requirements.txt`

***
