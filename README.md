Customer Lifetime Value (CLV) Prediction App
This Streamlit web application predicts 1-Year Customer Lifetime Value (CLV) based on customer purchasing behavior. The tool provides both manual input and bulk prediction via file upload, with rich visualizations and regional insights.

🔗 Live App: https://clvproject-g4fevnrrvwmx3dpshvqvmw.streamlit.app/

🚀 Features
🔍 Predict 1-Year CLV for individual customers via manual input

📁 Upload raw transaction data for bulk CLV forecasting

📊 Auto-segment customers into Low, Medium, High, Very High CLV groups

🌍 Analyze top countries by average CLV

📈 Visualize Quantity vs CLV segmented by customer groups

📥 Download prediction results as a CSV

💡 Clean, modern UI with custom styling for a better experience


🧠 How It Works
The app uses a regression model trained on customer transaction data to predict their expected value over the next year. Features used for prediction:

invoice_count: Number of unique invoices

total_quantity: Total quantity purchased

avg_price: Average unit price

recency_days: Days since the last purchase

lifespan_days: Customer tenure in days

📁 Input File Format (for Bulk Upload)
Upload a CSV or Excel file with the same format as the sample csv file provided.


🙋‍♂️ Author
Ayush
📧 Contact: ayushchutani633@gmail.com








Ask ChatGPT
