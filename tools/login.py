from numerapi import NumerAPI
from dotenv import load_dotenv
import os

def login() -> NumerAPI:
    load_dotenv()
    public_id = os.getenv("NUMERAI_PUBLIC_ID")
    secret_key = os.getenv("NUMERAI_SECRET_KEY")

    napi = NumerAPI(public_id, secret_key)

    account_info = napi.get_account()
    current_round = napi.get_current_round()
    print(f"Account_info: {account_info['username']}")
    print(f"Current_round: {current_round}")

    return napi
