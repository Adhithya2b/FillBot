# FillBot - Intelligent Form Filler

FillBot is an intelligent form-filling bot that uses semantic matching to automatically fill out web forms. It leverages natural language processing to understand form questions and match them with appropriate user data.

## Features

- Semantic matching of form questions with user data
- Support for various form field types:
  - Text inputs
  - Radio buttons
  - Checkboxes
  - Dropdown menus
  - Text areas
- Intelligent field type detection
- Robust error handling and retry mechanisms
- Chrome WebDriver integration

## Prerequisites

- Python 3.8 or higher
- Google Chrome browser
- Chrome WebDriver (automatically managed by webdriver-manager)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FillBot.git
cd FillBot
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `user_data.json` file with your form data. Example structure:
```json
{
    "Full Name": "John Doe",
    "Email": "john.doe@example.com",
    "Phone Number": "123-456-7890",
    "Date of Birth": "01/01/1990",
    "Address": "123 Main St, City, State"
}
```

## Usage

1. Run the bot:
```bash
python theBot.py
```

2. When prompted, enter the URL of the form you want to fill out.

## How It Works

1. The bot loads a pre-trained semantic model (all-MiniLM-L6-v2) for understanding text similarity
2. It creates embeddings for all field names in your user data
3. When presented with a form question, it uses semantic matching to find the most appropriate field from your data
4. The bot automatically detects field types and uses appropriate filling methods
5. It handles various edge cases and provides feedback on the filling process

## Security

- The current version doesn't require any API keys
- If you add external API integrations in the future:
  - Use environment variables for API keys
  - Never commit API keys to version control
  - Use `.env` file for local development

## Error Handling

The bot includes robust error handling for:
- Network issues
- Missing elements
- Invalid input formats
- Timeout scenarios

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is intended for legitimate use cases such as automating repetitive form filling tasks. Please use responsibly and in accordance with the terms of service of the websites you interact with. 
