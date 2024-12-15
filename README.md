# Flask Data Display Application

This Flask application displays extracted data in a categorized format.

## Setup Instructions

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask application:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

## Features

- Displays categorized data with Name, Address, Phone Number, Type of Information, and Rating
- Responsive design for better viewing on different devices
- Reads data from Excel file (extracted_data_ko.xlsx) or uses sample data if file not found

## File Structure

- `app.py`: Main Flask application
- `templates/index.html`: HTML template for data display
- `requirements.txt`: List of Python dependencies
- `extracted_data_ko.xlsx`: Data file (generated from image processing)
# translate-flask
