from schwab.auth import client_from_manual_flow
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_KEY, APP_SECRET = os.getenv('API_KEY'), os.getenv('APP_SECRET')
CALLBACK_URL = 'https://127.0.0.1'  
TOKEN_PATH = 'token.json'

def main():
    if not API_KEY or not APP_SECRET:
        print("env missing info")
        return
    
    return client_from_manual_flow(API_KEY, APP_SECRET, CALLBACK_URL, TOKEN_PATH)


if __name__ == '__main__':
    main()

