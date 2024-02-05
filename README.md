# Streamlit Bedrock SDXL Example

# Pre-requisites

- Python 3.10+
- awscli

# Setup

visit [Amazon Bedrock Console Page](https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/modelaccess) and enable the SDXL 1.0 model.

install the required packages using the following command:

```bash
$ pip install -r requirements.txt
```

copy [dev.env](/env/dev.env) to `.env` and fill in the required values.

```bash
$ cp env/dev.env .env
```

# Run

```bash
$ streamlit run app.py
```
