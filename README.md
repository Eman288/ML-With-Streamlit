# ML-With-Streamlit  

This repository contains a collection of machine learning models deployed using **Streamlit**, a Python-based framework for building interactive web applications. The project demonstrates how to effectively integrate machine learning workflows with a user-friendly interface, making it easier for end users to interact with and understand machine learning predictions.

---

## Features  
- **Streamlit-Powered Interface**: An intuitive and interactive web application.  
- **Pre-Trained Models**: Includes popular ML models such as classification, regression, or clustering. 
- **Real-Time Predictions**: Users can upload datasets or input values for real-time predictions.  
- **Data Visualization**: Interactive visualizations for datasets and models insights.  
---

## Technologies Used  
- **Python 3.8+**  
- **Streamlit**  
- **scikit-learn**  
- **pandas**
- **numpy**  
- **matplotlib/seaborn**  

---

## Installation  
To run this project locally, follow these steps:  

1. Clone the repository:  
   ```bash  
   git clone https://github.com/Eman288/ML-With-Streamlit.git  
   cd ML-With-Streamlit  
   ```  

2. Create a virtual environment:  
   ```bash  
   python -m venv environmentname  
   source environmentname/bin/activate # On Windows: environmentname\Scripts\activate  
   ```  
3. Activate the virtual environment:
   ```bash
    environmentname\scripts\activate
   ```

5. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

6. Run the Streamlit app:  
   ```bash  
   streamlit run app.py  
   ```  

7. Open your browser and navigate to:  
   ```text  
   http://localhost:8501  
   ```  

---

## Project Structure  
```plaintext  
ML-With-Streamlit/  
├── app.py                # Main Streamlit application file  
├── requirements.txt      # Python dependencies  
├── cleaning.ipynb        # the file used to clean the alzheimers dataset
├── prediction.ipynb      # the file used to create the model and starts the prediction
├── README.md             # Project documentation
```
---

## How to Use  
1. **Upload Dataset**: Upload your dataset in csv or xlsx format via the app interface.  
2. **Select Model**: Choose from the available machine learning models.  
3. **Set Parameters**: Adjust model-specific parameters.  
4. **View Results**: See predictions, evaluation metrics, and visualizations in real time.  

 
---
## Contributing  

- Eman Tamam: [Eman288](https://github.com/Eman288)  
- Amna Abd-Elrazek: [amnaabdelrazek](https://github.com/amnaabdelrazek)
- Arwa Mostafa: [ArwaMostafa19](https://github.com/ArwaMostafa19)
- Arwa Khaled: [arwakhaled123](https://github.com/arwakhaled123)
- Yasmeen Khaled: [YasmeenFci](https://github.com/YasmeenFci)

---

## License  
This project is licensed under the **MIT License**. See the `LICENSE` file for details.  
