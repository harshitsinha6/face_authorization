
import os
import face_recognition


def get_known_faces():
    known_faces_encodings = {}
    for known_face in os.listdir("known_faces"):
        curr_person_encoding = []
        for face in os.listdir(f"known_faces/{known_face}"):
            curr_face = face_recognition.load_image_file(f"known_faces/{known_face}/{face}")
            curr_person_encoding.append(
                face_recognition.face_encodings(curr_face)
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
    return x1 < x < x2 and y1 < y < y2

