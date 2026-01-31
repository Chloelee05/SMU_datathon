# SMU DATATHON

A modern, clean ChatGPT-style chatbot interface built with **Python Streamlit**.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Screenshots


## Installation

1. **Clone the repository**
```bash
git clone https://github.com/Chloelee05/SMU_datathon/
cd chatgpt-clone-ui
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set OPENAI_API_KEY**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open in browser**
```
http://localhost:8501
```

## Project Structure

```
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## Usage

### Starting a New Chat
Click the "✏️ New chat" button in the sidebar to start a fresh conversation.

### Managing Chat History
- Click on any chat in the "Recent" section to load it
- Click the 🗑️ button next to a chat to delete it

### Accessing Profile & Settings
Click "👤 Lee Chloe" at the bottom of the sidebar to open the profile modal with access to:
- ⚙️ Settings
- 🎨 Theme
- 📊 Usage
- ❓ Help

## Customization

### Connecting to OpenAI API
To enable real AI responses, modify the chat response section in `app.py`:

```python
import openai

openai.api_key = "your-api-key"

# Replace the demo response with:
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}]
)
```

### Changing Theme Colors
Edit the CSS variables in the `st.markdown()` section of `app.py` to customize colors.

## Tech Stack

- **Frontend**: Streamlit
- **Styling**: Custom CSS
- **Language**: Python 3.8+

## Requirements

- Python 3.8 or higher
- Streamlit 1.28.0 or higher

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by [OpenAI ChatGPT](https://chat.openai.com)
- Built with [Streamlit](https://streamlit.io)
