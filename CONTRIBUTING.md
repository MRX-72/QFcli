# Contributing to QFcli

Thank you for considering contributing to QFcli! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version)

### Suggesting Features

We love new ideas! Please open an issue describing:
- The feature you'd like to see
- Why it would be useful
- How it might work

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test your changes thoroughly
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and modular

### Testing

Please ensure your changes:
- Don't break existing functionality
- Work with multiple stock tickers
- Handle edge cases gracefully

## Development Setup

```bash
# Clone your fork
git clone https://github.com/MRX-72/QFcli.git
cd QFcli

# Install dependencies
pip install -r requirements.txt

# Test your changes
python main.py AAPL
python main.py --compare AAPL MSFT
```

## Questions?

Feel free to open an issue for any questions or clarifications.

Thank you for contributing! 🎉
