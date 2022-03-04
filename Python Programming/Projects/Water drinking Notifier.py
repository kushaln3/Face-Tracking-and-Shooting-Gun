from plyer import notification

notification.notify(
    title='Please drink water',
    message='Please drink water within 10s in front of your computer with your webcam on or else your computer will be shutdown, This is done to force you to drink water',
    timeout=60
)