#============================================#
# main.py                                    #
# - 이 파일을 실행하면 모든 과정이 진행된다. #
#============================================#


#=========#
# Modules #
#=========#
import cv2
import os
import numpy as np
import picamera
from picamera.array import PiRGBArray
import sys
import time
import RPi.GPIO as GPIO
#from video import create_capture
#from common import clock, draw_str


#==================#
# Global Variables #
#==================#
PERSONS = ['', 'I-RENE', 'Hyun ji', 'So hye']


#===========#
# Functions #
#===========#
# 1) 얼굴 감지 함수
# - OpenCV를 이용해 얼굴을 감지함
def detect_face(img):
    # 테스트 이미지를 회색 배경으로 변환한다. opencv 얼굴 감지 모듈이 회색만 인지하기 때문
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # OpenCV 얼굴 감지기를 불러온다. LBP를 사용한다.
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    # 다중스케일 이미지들을 감지. 결과물은 얼굴 리스트로 반환된다.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    # 감지된 얼굴이 없다면 원래 이미지를 반환한다.
    if len(faces) == 0:
        return None, None
    # 얼굴이 하나 있을 거라는 가정 하에, 얼굴 영역을 추출한다.
    (x, y, w, h) = faces[0]
    # 이미지의 얼굴 부분만을 반환한다.
    return gray[y:y+w, x:x+h], faces[0]


# 2) 데이터 학습 준비 함수
# - 모든 사람의 트레이닝 이미지를 읽고, 각 이미지로부터 얼굴을 감지한다.
# - 같은 길이의 두 리스트를 반환한다.
# - 하나는 얼굴 리스트, 나머지 하나는 각 얼굴의 레이블 리스트이다.
def prepare_training_data(data_folder_path):
    #---------------#
    #-----1단계-----#
    #---------------#
    # 데이터 폴더 내부의 디렉토리들을 얻어온다.(각 주체마다 하나의 디렉토리)
    dirs = os.listdir(data_folder_path)
    # 모든 인물들의 얼굴들을 리스트로 모은다.
    faces = []
    # 모든 인물들의 레이블을 리스트로 모은다.
    labels = []
    # 각 디렉토리를 순회하면서 이미지를 불러와야 한다.
    for dir_name in dirs:
        # 디렉토리명의 첫 글자가 s여야 한다. 따라서 아닌 것들은 제외시킨다.
        if not dir_name.startswith('s'):
            continue
        #---------------#
        #-----2단계-----#
        #---------------#
        # s 뒤에 숫자가 붙을 것이다. 그 숫자만 색출해내기 위해 s는 없앤다.
        # 그리고 제거된 데이터를 정수 데이터로 변환하여 레이블에 저장한다.
        label = int(dir_name.replace('s', ''))
        # 학습시킬 데이터는 training-data라는 이름의 디렉토리에 있다.
        # 그리고 각 인물들의 디렉토리 경로는 s1, s2, s3...등으로 저장된다.
        # 즉, "training-data/s1", "training-data/s2"로 변환해야 한다.
        person_dir_path = data_folder_path + '/' + dir_name
        # 주어진 인물 디렉토리 안에 있는 이미지들의 이름들을 얻는다.
        person_images_names = os.listdir(person_dir_path)
        #---------------#
        #-----3단계-----#
        #---------------#
        # 각 이미지 이름을 통해 이미지를 읽고, 얼굴을 인식하여 얼굴 리스트에 얼굴을 추가한다.
        for image_name in person_images_names:
            # .로 시작하는 이미지. 즉, 숨김 파일은 제외시킨다.
            if image_name.startswith('.'):
                continue
            # 이미지 경로를 빌드한다.
            # 예시 경로 = training-data/s1/1.png
            image_path = person_dir_path + '/' + image_name
            # 이미지를 읽어온다.
            image = cv2.imread(image_path)
            # 이미지를 보여주기 위해 이미지 윈도우를 표시한다.
            cv2.imshow('Training on images...', cv2.resize(image, (400, 500)))
            cv2.waitKey(100) # 매개변수는 대기 시간이다.
            # 얼굴을 감지한다.
            face, rect = detect_face(image)
            #---------------#
            #-----4단계-----#
            #---------------#
            # 감지되지 않은 얼굴들은 전부 제외시킨다.
            if face is not None:
                # faces 리스트에 얼굴을 추가한다.
                faces.append(face)
                # labels 리스트에 얼굴을 추가한다.
                labels.append(label)
    # 모든 창을 제거한다.
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    # 얼굴과 레이블에 관련된 faces, labels 리스트를 쌍으로 반환한다.
    return faces, labels


# 3) 직사각형 그리기 함수
# - 이미지에 직사각형을 그리는 함수이다.
# - x, y 좌표를 통해 width, height를 주어 그린다.
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) # 초록색


# 4) 텍스트 그리기 함수
# - 이미지의 x, y 좌표가 시작되는 곳에 텍스트를 그리는 함수이다.
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2) # 초록색


# 5) 예측 함수
# - 이미지가 지나갈 때 그 안의 사람을 인식하고,
# - 감지된 얼굴에 해당하는 사람의 이름을 직사각형 주위에 그려주는 함수이다.
def predict(test_img):
    global face_recognizer
    global PERSONS
    # 원본 이미지를 변경하고 싶지 않으므로 복사본을 만든다.
    img = test_img.copy()
    # 이미지로부터 얼굴을 감지한다.
    face, rect = detect_face(img)
    # 얼굴 인식기를 이용해 이미지를 예측한다.
    label, confidence = face_recognizer.predict(face)
    # 얼굴 인식기로부터 반환된 레이블을 텍스트로 지정한다.
    label_text = PERSONS[label]
    # 감지된 얼굴 주위에 직사각형을 그린다.
    draw_rectangle(img, rect)
    # 예측된 사람의 이름을 그린다.
    draw_text(img, label_text, rect[0], rect[1]-5)
    return img

#-----------#
# main 함수 #
#-----------#
def main():
    global face_recognizer
    '''
    1. 사진 폴더에 있는 이미지들을 가져온다.
    '''
    # 비교에 필요한 데이터를 준비하자.
    print('Preparing data...')
    # 필요한 인원수는 우선 동아리 6명이므로,
    # 6개의 디렉터리에 해당하는 만큼 준비해야 한다.
    faces, labels = prepare_training_data('training-data')
    #faces1, labels1 = prepare_training_data('training-data1')
    #faces2, labels2 = prepare_training_data('training-data2')
    #faces3, labels3 = prepare_training_data('training-data3')
    #faces4, labels4 = prepare_training_data('training-data4')
    # 데이터 준비 완료
    print('Data prepared')
    # 각 데이터의 감지된 수를 출력해준다.
    print('Total faces: ', len(faces))
    print('Total labels: ', len(labels))
    '''
    print('Total faces1: ', len(faces1))
    print('Total labels1: ', len(labels1))
    print('Total faces2: ', len(faces2))
    print('Total labels2: ', len(labels2))
    print('Total faces3: ', len(faces3))
    print('Total labels3: ', len(labels3))
    print('Total faces4: ', len(faces4))
    print('Total labels4: ', len(labels4))
    '''
    '''
    2. 가져온 이미지들을 학습시킨다.
    '''
    #-------------------------#
    # 얼굴 인식 트레이닝 시작 #
    #-------------------------#
    # 세 가지 선택지가 있다.
    # 1. EigenFace
    # 2. FisherFace
    # 3. Local Binary Patterns Histogram (LBPH)
    # 이 중 3번 선택지를 활용할 것이다.
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    # labels의 경우 numpy 모듈의 array를 통해 저장하여 학습시킨다.
    face_recognizer.train(faces, np.array(labels))
    #face_recognizer.train(faces1, np.array(labels1))
    #face_recognizer.train(faces2, np.array(labels2))
    #face_recognizer.train(faces3, np.array(labels3))
    #face_recognizer.train(faces4, np.array(labels4))
    '''
    3. 사진을 찍는다.
    - test-data 디렉터리 안에,
    - 각각 test1, test2, test3, test4 이라는 이름으로 저장해야 한다.
    '''
    #camera = picamera.PiCamera()
    #count = 1
    #camera.capture('/home/pi/github/I-CAST/test-data.testasasdsc')
    camera = picamera.PiCamera()
    camera.start_preview()
    cv2.waitKey(1000)
    camera.resolution = (640, 480)
    camera.capture('test-data/test2.jpg')
    camera.stop_preview()
    
    # count = 1 # 사람 수 세기 변수
    #while count < 5:
    #    s = 'test-data/test'
    #    s += '%d' % count
    #    s += '.jpg'
    #   camera.capture(s)
    #     if cv2.waitKey(30) == 27:
    #       count += 1
    #        continue

    '''
    4. 찍은 사진을 저장한다.
    '''
    '''
    5. 그 사진을 비교 데이터로 선정한다.
    '''
    '''
    6. 학습된 데이터와 비교한다.
    '''
    '''
    7. 일치하면 해당 자료가 일치한다는 사실을 보여준다.
    '''
    '''
    8. 일치하지 않으면 오류음을 출력한다.
    '''
    try:
        print('Predicting images...')
        # 테스트 이미지들을 불러온다.
        test_img1 = cv2.imread('test-data/test1.jpg')
        test_img2 = cv2.imread('test-data/test2.jpg')
        test_img3 = cv2.imread('test-data/test3.jpg')
        #test_img4 = cv2.imread('test-data/test4.jpg')
        # 예측을 수행한다.
        predicted_img1 = predict(test_img1)
        predicted_img2 = predict(test_img2)
        predicted_img3 = predict(test_img3)
        #predicted_img4 = predict(test_img4)
        print('Prediction complete')
        # 모든 이미지들을 표시한다.
        cv2.imshow(PERSONS[1], cv2.resize(predicted_img1, (400, 500)))
        cv2.imshow(PERSONS[2], cv2.resize(predicted_img2, (400, 500)))
        cv2.imshow(PERSONS[3], cv2.resize(predicted_img3, (400, 500)))
        os.system('omxplayer correct.wav')
        #cv2.imshow(PERSONS[4], cv2.resize(predicted_img4, (400, 500)))
    except cv2.error:
        os.system('omxplayer incorrect.wav')
    finally:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()


#====================#
# 프로그램 실행 부분 #
#====================#
if __name__ == '__main__':
    main()

