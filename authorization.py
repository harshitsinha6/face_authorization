
import cv2
from datetime import datetime
from logging import log
import face_recognition
import util
import face_detection_yolo
import env_variables.environment_variables as app_envs


class Authorization:

    def __init__(self, video_source):
        self.vdo_src = video_source
        self.known_faces = None
        self.face_match_boundary = (150, 80, 300, 320)
        self.latest_checkin_time = {}
        self.latest_checkout_time = {}

    def get_known_faces(self):
        self.known_faces = util.get_known_faces()

    def detect(self, frame):

        _, face_locations = face_detection_yolo.get_face_locations(frame)

        for loc in face_locations:
            frame = cv2.rectangle(frame, (loc[0], loc[1]), (loc[2], loc[3]), (255, 255, 0), 2)

        face_to_check = []
        while face_locations:
            face = face_locations.pop(0)
            if util.is_inside(self.face_match_boundary, face):
                face_to_check.append(face)

        # print(f"face to match: {face_to_check}")
        face_encodings = face_recognition.face_encodings(frame)  # , face_to_check)

        tracking_list = []
        for known_face, known_face_encodings in self.known_faces.items():

            matched_counter, total_counter = 0, 0
            for known_face_encoding in known_face_encodings:
                for face_encoding in face_encodings:
                    try:
                        if face_recognition.compare_faces([known_face_encoding], face_encoding)[0]:
                            matched_counter += 1
                        total_counter += 1
                    except Exception as e:
                        print(e)
            if total_counter > 0:
                tracking_list.append((matched_counter/total_counter, known_face))

        tracking_list.sort(reverse=True)
        print(tracking_list)
        return tracking_list[0][1] if tracking_list else "unknown"

    def start_stream(self):

        self.get_known_faces()

        # opening the camera
        cap = cv2.VideoCapture(self.vdo_src)

        while True:
            ret, frame = cap.read()
            if not ret:
                log.debug(f"unable to get the frames for: {self.vdo_src}")
                continue

            potential_match = self.detect(frame)
            # frame, _ = face_detection_yolo.get_face_locations(frame)

            if potential_match != "unknown":

                checked_time = datetime.now()
                self.latest_checkin_time[potential_match] = checked_time.strftime("%d/%m/%Y %H:%M:%S")
                print(self.latest_checkin_time)

                util.post_details(potential_match, checked_time.strftime("%d/%m/%Y %H:%M:%S"), app_envs.CHECKED_TYPE)

            cv2.putText(frame, f"{potential_match}", (self.face_match_boundary[0]-20, self.face_match_boundary[1]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.rectangle(frame, (self.face_match_boundary[0], self.face_match_boundary[1]),
                          (self.face_match_boundary[0] + self.face_match_boundary[2],
                          self.face_match_boundary[1] + self.face_match_boundary[3]), (0, 255, 0), 2)
            # to show
            cv2.imshow("__SCREEN__", frame)
            # cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



