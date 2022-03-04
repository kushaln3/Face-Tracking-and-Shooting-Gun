#include <Servo.h>
  
Servo servo1;
Servo servo2;
Servo servo3;
int joyX = 0;
int joyY = 1;
int joySWPin = 1;
int servoVal;

void setup() 
{
  servo1.attach(9); //x axis
  servo2.attach(10); //y axis
  servo3.attach(11); //trigger
}
  
void loop()
{
    if (digitalRead(joySWPin) == 0){
    servo3.write(179);
    delay(250);
    }
    else{
    servoVal = analogRead(joyX);
    servoVal = map(servoVal, 0, 1023, 0, 180);
    servo1.write(servoVal);

    servoVal = analogRead(joyY);
    servoVal = map(servoVal, 0, 1023, 0, 180);
    servo2.write(servoVal);
    delay(15);
    }

}
