import cv2
import numpy as np
import poseEstimationModule as pem 


EXERCISES = {
    "1": {"name": "Left Bicep Curl", "points": [11, 13, 15], "range": (210, 310)},
    "2": {"name": "Right Bicep Curl", "points": [12, 14, 16], "range": (160, 30)},
    "3": {"name": "Squats", "points": [24, 26, 28], "range": (170, 70)},
    "4": {"name": "Shoulder Press", "points": [12, 14, 16], "range": (90, 170)},
    "5": {"name": "Push-Ups", "points": [12, 14, 16], "range": (80, 170)},
    "6": {"name": "Lunges", "points": [24, 26, 28], "range": (170, 90)},
    "7": {"name": "Sit-Ups", "points": [12, 24, 26], "range": (170, 50)}
}

class Button:
    def __init__(self, pos, text, scale=3, thickness=3, color=(255, 0, 255), text_color=(255, 255, 255)):
        self.pos = pos
        self.text = text
        self.scale = scale
        self.thickness = thickness
        self.color = color
        self.text_color = text_color
        (self.w, self.h), _ = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_PLAIN, self.scale, self.thickness)
        self.x1, self.y1 = self.pos
        self.x2, self.y2 = self.x1 + self.w + 20, self.y1 + self.h + 20 

    def draw(self, img):
        cv2.rectangle(img, (self.x1, self.y1), (self.x2, self.y2), self.color, cv2.FILLED)
        cv2.putText(img, self.text, (self.x1 + 10, self.y1 + self.h + 10), 
                    cv2.FONT_HERSHEY_PLAIN, self.scale, self.text_color, self.thickness)

    def is_clicked(self, x, y):
        if self.x1 < x < self.x2 and self.y1 < y < self.y2:
            return True
        return False


current_state = "MENU" 
selected_exercise = None
click_pos = None

def mouse_click(event, x, y, flag, param):
    global click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        click_pos = (x, y)

def main():
    global current_state, selected_exercise, click_pos
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)  
    
    detect = pem.PoseEstimation()
    
    cv2.namedWindow("AI Trainer")
    cv2.setMouseCallback("AI Trainer", mouse_click)

    buttons = []
    y_start = 150
    keys = list(EXERCISES.keys())

    for i, key in enumerate(keys):
        x_pos = 100 if i < 4 else 650
        y_pos = y_start + (i % 4) * 100
        btn = Button((x_pos, y_pos), EXERCISES[key]["name"], scale=2)
        btn.exercise_key = key 
        buttons.append(btn)
        
    back_btn = Button((50, 50), "< Back", scale=2, color=(0, 0, 255))

    while True:
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)

        if current_state == "MENU":
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (1280, 720), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            
            cv2.putText(img, "SELECT AN EXERCISE", (400, 80), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

            for btn in buttons:
                btn.draw(img)
                if click_pos:
                    if btn.is_clicked(click_pos[0], click_pos[1]):
                        selected_exercise = EXERCISES[btn.exercise_key]
                        current_state = "EXERCISE"
                        click_pos = None 

        elif current_state == "EXERCISE":
            img = detect.findPose(img, False)
            lmList = detect.positions(img, False)
            
            back_btn.draw(img)
            if click_pos:
                if back_btn.is_clicked(click_pos[0], click_pos[1]):
                    current_state = "MENU"
                    click_pos = None
            
            if len(lmList) != 0:
                points = selected_exercise["points"]
                r = selected_exercise["range"]
                angle = detect.findAngle(img, points[0], points[1], points[2])
                per = np.interp(angle, (r[0], r[1]), (0, 100))
                bar = np.interp(angle, (r[0], r[1]), (650, 100))
                color = (0, 255, 0)
                if per == 100: color = (0, 255, 0)
                if per == 0: color = (0, 0, 255)
                
                cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
                cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
                cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)
                
                cv2.putText(img, selected_exercise["name"], (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("AI Trainer", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()