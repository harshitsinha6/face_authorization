
import os
import face_recognition


def get_known_faces():
    known_faces_encodings = {}
    for known_face in os.listdir("known_faces"):
        curr_person_encoding = []
        for face in os.listdir(f"known_faces/{known_face}"):
            curr_face = face_recognition.load_image_file(f"known_faces/{known_face}/{face}")
            curr_person_encoding.append(
                face_recognition.face_encodings(curr_face)[0]
            )
        known_faces_encodings[known_face] = curr_person_encoding
        # print(known_face, len(curr_person_encoding))
    return known_faces_encodings

# get_known_faces()


def match_face(known_face_encoding, face_encoding):
    total = 0
    matched = 0
    for known_image_harshit in known_face_encoding:
        if face_recognition.compare_faces([known_image_harshit], face_encoding):
            matched += 1
        total += 1
    return matched / total


def is_inside(rect1, rect2):
    x1, y1, x2, y2 = rect1
    p, q, r, s = rect2
    if findPoint(x1, y1, x2+x1, y1+y2, p, q) and findPoint(x1, y1, x2+x1, y1+y2, r, s):
        return True
    else:
        return False


def findPoint(x1, y1, x2, y2, x, y):
    return (x > x1 and y > y1) and (x < x2 and y < y2)
    # return x1 < x < x2 and y1 < y < y2


def post_details(potential_match, match_time, checked_type):
    # make api request with potential_match, its time and its checked_type
    pass


def get_snapshot_and_save():
    name = input("Enter Name: ")
    number_of_snapshots = 5

    import cv2
    cap = cv2.VideoCapture(0)

    os.mkdir(f"known_faces/{name}")

    while number_of_snapshots > 0:
        ret, frame = cap.read()

        cv2.imshow("frame", frame)
        cv2.waitKey(1)
        v = input("want to save? (0 - no / 1 - yes)")
        if v == '1':
            number_of_snapshots -= 1

            cv2.imwrite(f"known_faces/{name}/image{number_of_snapshots}.jpg", frame)

    cap.release()
    cv2.destroyAllWindows()



