# Face ID Recognition System

A modern and efficient face recognition-based attendance system with real-time updates and a beautiful user interface.

## Features

- Real-time face recognition
- Automatic attendance marking
- Modern and responsive UI
- Department-based filtering
- Real-time attendance updates
- Export functionality (CSV/PDF)
- Search and filter capabilities
- Secure face registration

## Prerequisites

- Python 3.7 or higher
- Webcam
- Internet connection (for CDN resources)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd new_ch_face_recognition
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Register new faces:
   - Go to the Register page
   - Fill in the required information
   - Capture your face using the webcam
   - Click "Capture & Register"

4. View attendance:
   - Go to the Attendance page
   - Use the search and filter options to find specific records
   - Export data using the export buttons

## System Requirements

- Operating System: Windows 10/11, macOS, or Linux
- RAM: Minimum 4GB (8GB recommended)
- Webcam: 720p or higher resolution
- Browser: Chrome, Firefox, or Edge (latest versions)

## Security Features

- Face data is stored locally
- No personal data is shared with third parties
- Secure face encoding and matching
- Automatic attendance interval to prevent duplicates

## Troubleshooting

If you encounter any issues:

1. Ensure your webcam is properly connected and accessible
2. Check that all dependencies are installed correctly
3. Make sure you have the latest version of your web browser
4. Verify that your system meets the minimum requirements

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 