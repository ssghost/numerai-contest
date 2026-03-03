from login import login
from constants import CURR_MODEL, RESULT_PATH

def upload() -> None:
    try:
        napi = login()
        models = napi.get_models()

        if CURR_MODEL in models:
            model_id = models[CURR_MODEL]
            print(f"Found model name: {CURR_MODEL}, ID: {model_id}")
            submission_id = napi.upload_predictions(RESULT_PATH, model_id=model_id)
            print(f"Submission ID: {submission_id}")
        else:
            print(f"Can't find model with name'{CURR_MODEL}'.")

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == '__main__':
    upload()