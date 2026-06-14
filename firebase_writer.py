"""
Firebase writer/reader using REST API.

This version does NOT use firebase-admin.
It avoids protobuf / mediapipe dependency conflicts.

Requirement:
Realtime Database Rules for demo:
{
  "rules": {
    ".read": true,
    ".write": true
  }
}
"""

from datetime import datetime
import requests


DATABASE_URL = "https://eye-gaze-nurse-call-default-rtdb.asia-southeast1.firebasedatabase.app"


PRIORITY_MAP = {
    "Call Nurse": "High",
    "Emergency": "High",
    "I am in pain": "High",
    "I need water": "Low",
    "I need restroom assistance": "Low",
    "I feel uncomfortable": "Low",
}


def firebase_url(path):
    path = path.strip("/")
    return f"{DATABASE_URL}/{path}.json"


def get_next_request_id():
    counter_url = firebase_url("counter/request_id")

    response = requests.get(counter_url, timeout=5)

    if response.status_code != 200:
        raise RuntimeError(f"Cannot read counter: {response.text}")

    current_id = response.json()

    if current_id is None:
        current_id = 0

    next_id = int(current_id) + 1

    response = requests.put(counter_url, json=next_id, timeout=5)

    if response.status_code != 200:
        raise RuntimeError(f"Cannot update counter: {response.text}")

    return next_id


def send_request_to_firebase(request_type):
    request_id = get_next_request_id()

    request_data = {
        "request_id": request_id,
        "type": request_type,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "priority": PRIORITY_MAP.get(request_type, "Low"),
        "status": "Pending"
    }

    current_response = requests.put(
        firebase_url("current_request"),
        json=request_data,
        timeout=5
    )

    if current_response.status_code != 200:
        raise RuntimeError(f"Cannot write current_request: {current_response.text}")

    history_response = requests.put(
        firebase_url(f"request_history/req_{request_id}"),
        json=request_data,
        timeout=5
    )

    if history_response.status_code != 200:
        raise RuntimeError(f"Cannot write request_history: {history_response.text}")

    print("\n[FIREBASE REQUEST SENT]")
    print(request_data)

    return request_data


def get_current_request_from_firebase():
    response = requests.get(firebase_url("current_request"), timeout=5)

    if response.status_code != 200:
        raise RuntimeError(f"Cannot read current_request: {response.text}")

    return response.json()


def get_current_status_from_firebase(expected_request_id=None):
    """
    Read current_request/status from Firebase.
    If expected_request_id is given, only return status when current_request has same ID.
    This prevents old/cleared request status from affecting patient screen.
    """
    data = get_current_request_from_firebase()

    if not data:
        return None

    if expected_request_id is not None:
        try:
            firebase_id = int(data.get("request_id", -1))
            expected_id = int(expected_request_id)
            if firebase_id != expected_id:
                return None
        except Exception:
            return None

    return data.get("status", None)


def update_current_status_in_firebase(new_status):
    response = requests.put(
        firebase_url("current_request/status"),
        json=new_status,
        timeout=5
    )

    if response.status_code != 200:
        raise RuntimeError(f"Cannot update status: {response.text}")

    return new_status


if __name__ == "__main__":
    send_request_to_firebase("I need water")