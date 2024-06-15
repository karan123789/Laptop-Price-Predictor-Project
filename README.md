# Laptop-Price-Predictor-Project

This code begins by importing necessary libraries such as NumPy for numerical computations, Pandas for data manipulation, Matplotlib for visualization, and Scikit-Learn for machine learning algorithms. The dataset containing information about various laptop features and their prices is loaded into a Pandas DataFrame. Data preprocessing steps like handling missing values, encoding categorical variables, and splitting the dataset into training and testing sets are performed.

Next, a machine learning pipeline is set up using Scikit-Learn, where a Decision Tree algorithm is chosen for its simplicity and interpretability in predicting laptop prices based on features like processor speed, RAM, storage capacity, etc. The model is trained using the training data and then evaluated using the testing data to assess its predictive performance.

Once the model is trained and evaluated, it can be serialized using Pickle for future use without needing to retrain the model every time. Finally, Matplotlib is used to visualize the predicted laptop prices compared to the actual prices, providing insights into the model's accuracy and potential areas for improvement.



https://laptop-price-predictor-project-td6jwpscsraztrrzwhewvl.streamlit.app/



NOTE: Prices are around reasonable however further testing is required to make it's accuracy better
