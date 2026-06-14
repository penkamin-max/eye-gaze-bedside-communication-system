\# Eye-Gaze Bedside Communication System



\## Overview



This project presents a low-cost eye-gaze bedside communication system designed for patients with severe or temporary motor limitations.



The patient uses eye gaze to select predefined assistance requests without pressing physical buttons. Requests are transmitted through Firebase Realtime Database and displayed on a mobile-friendly caregiver dashboard.



The caregiver can acknowledge or complete each request, and the updated status is displayed back on the patient-side interface.



\## Main Features



\* Webcam-based eye-gaze interaction

\* MediaPipe Face Mesh and iris landmark detection

\* OpenCV real-time frame processing

\* 10-frame moving-average filtering

\* Two-choice left/right paging interface

\* 1.36-second dwell selection

\* Firebase Realtime Database communication

\* Mobile caregiver dashboard using Firebase Hosting

\* Request history and status updates

\* Local JSON backup during network failure



\## Supported Patient Requests



\* Call Nurse

\* Emergency

\* I am in pain

\* I need water

\* I need restroom assistance

\* I feel uncomfortable



\## Request Status Flow



```text

Pending → Acknowledged → Completed

```



\## System Architecture



```text

Webcam

&#x20;  ↓

OpenCV Frame Capture

&#x20;  ↓

MediaPipe Face Mesh and Iris Landmarks

&#x20;  ↓

Gaze Estimation

&#x20;  ↓

Moving-Average Smoothing

&#x20;  ↓

Two-Choice Paging Interface

&#x20;  ↓

1.36-Second Dwell Confirmation

&#x20;  ↓

Firebase Realtime Database

&#x20;  ├── current\_request

&#x20;  └── request\_history

&#x20;  ↓

Mobile Caregiver Dashboard

&#x20;  ↓

Acknowledged / Completed Feedback

&#x20;  ↓

Patient-Side Visual Feedback

```



\## Project Structure



```text

DoAn\_BME/

│

├── gaze\_control.py

├── firebase\_writer.py

├── mobile\_dashboard.html

├── firebase.json

├── .firebaserc

├── .gitignore

├── requirements.txt

│

├── public/

│   └── index.html

│

└── gaze\_dataset/

```



\## Technologies



\* Python

\* OpenCV

\* MediaPipe

\* NumPy

\* Requests

\* HTML

\* CSS

\* JavaScript

\* Firebase Realtime Database

\* Firebase Hosting



\## Installation



Clone the repository:



```bash

git clone https://github.com/penkamin-max/eye-gaze-bedside-communication-system.git

```



Move into the project folder:



```bash

cd eye-gaze-bedside-communication-system

```



Create a virtual environment:



```bash

python -m venv .venv

```



Activate the virtual environment on Windows:



```powershell

.\\.venv\\Scripts\\activate

```



Install dependencies:



```bash

pip install -r requirements.txt

```



\## Running the Patient-Side Application



Run:



```powershell

python gaze\_control.py

```



Main controls:



```text

Q = Quit

P = Pause

R = Reset to START

F = Toggle fullscreen

C = Recalibrate center

X = Invert horizontal direction

Y = Invert vertical direction

```



Backup request keys:



```text

E = Emergency

W = I need water

N = Call Nurse

D = Clear current request

```



\## Caregiver Dashboard



The mobile caregiver dashboard is deployed through Firebase Hosting.



Dashboard link:



```text

https://eye-gaze-nurse-call.web.app/

```



The dashboard allows caregivers to:



\* View the current request

\* Review request history

\* Identify high-priority alerts

\* Acknowledge requests

\* Complete requests

\* Clear the current request



\## Firebase Data Structure



```text

counter/

&#x20;   request\_id



current\_request/

&#x20;   request\_id

&#x20;   type

&#x20;   timestamp

&#x20;   priority

&#x20;   status



request\_history/

&#x20;   req\_ID/

&#x20;       request\_id

&#x20;       type

&#x20;       timestamp

&#x20;       priority

&#x20;       status

```



\## Team Members



\### Phạm Đình Tuấn Minh



Responsibilities:



\* Patient-side gaze-control application

\* Firebase integration

\* Request-writing logic

\* Camera testing and debugging

\* Technical report



\### Nguyễn Phan Hà Minh



Responsibilities:



\* Mobile caregiver dashboard

\* Request display and status updates

\* Presentation preparation

\* Integration testing



\## Limitations



\* Gaze accuracy depends on lighting and camera placement

\* The current prototype mainly supports one patient

\* Internet access is required for Firebase communication

\* The system is not a certified medical device

\* Authentication and clinical-grade security are not yet implemented



\## Future Work



\* User testing with target patient groups

\* Firebase Authentication

\* Multi-patient and multi-bed support

\* Push notifications and enhanced alarms

\* Personalized gaze calibration

\* Embedded deployment using Raspberry Pi or similar devices



\## Disclaimer



This project is an educational prototype. It is not intended to replace certified medical nurse-call systems or professional healthcare equipment.



\## License



This project is developed for academic purposes at Hanoi University of Science and Technology.



